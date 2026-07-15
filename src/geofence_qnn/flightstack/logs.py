"""Turn real flight logs into the state-action datasets used by the pipeline.

Supported sources:

- PX4 Autopilot ULog (``.ulg``) via ``pyulog`` — topic
  ``vehicle_local_position`` by default (NED local frame).
- ArduPilot DataFlash (``.bin``/``.log``) and MAVLink telemetry (``.tlog``)
  via ``pymavlink`` — ``LOCAL_POSITION_NED`` for telemetry logs and
  ``XKF1``/``NKF1`` EKF estimates for DataFlash logs.
- Generic CSV with columns ``t,x,y,vx,vy[,ax,ay][,episode]`` so any stack
  (or ``ulog2csv``/``mavlogdump`` output, or the SITL recorder in this
  package) can feed the experiments without binary parsers.

All loaders return :class:`Trajectory` objects in the experiment's planar
world frame. NED logs are mapped to the world frame as ``x = east, y = north``
and an optional ``offset`` shifts the log positions so the recorded flights
line up with the experiment's geofence geometry.

Actions are the accelerations the flight stack actually commanded/achieved:
the logged acceleration when present, otherwise the forward finite difference
of the resampled velocity. They are clipped to ``amax`` and normalized the
same way as the synthetic teacher actions, so the rest of the pipeline
(training, quantization, E0-E2, Monte Carlo) is unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

from ..features import state_features
from ..geometry import ForbiddenBox


@dataclass
class Trajectory:
    """One continuous flight segment in world coordinates."""

    t: np.ndarray  # (n,) seconds, strictly increasing
    pos: np.ndarray  # (n, 2) world x, y
    vel: np.ndarray  # (n, 2) world vx, vy
    acc: np.ndarray | None = None  # (n, 2) commanded/measured acceleration, optional
    source: str = ""

    def __len__(self) -> int:
        return len(self.t)


def _apply_frame(
    xy_first: np.ndarray,
    xy_second: np.ndarray,
    frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map logged axes to world axes.

    ``ned`` logs store (north, east); the world frame is (x=east, y=north).
    ``xy`` logs are already in world axes.
    """
    if frame == "ned":
        return xy_second, xy_first
    if frame == "xy":
        return xy_first, xy_second
    raise ValueError(f"unknown frame: {frame!r} (expected 'ned' or 'xy')")


def _finalize_trajectory(
    t: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    vfirst: np.ndarray,
    vsecond: np.ndarray,
    afirst: np.ndarray | None,
    asecond: np.ndarray | None,
    frame: str,
    offset: tuple[float, float],
    source: str,
) -> Trajectory | None:
    x, y = _apply_frame(np.asarray(first, float), np.asarray(second, float), frame)
    vx, vy = _apply_frame(np.asarray(vfirst, float), np.asarray(vsecond, float), frame)
    t = np.asarray(t, dtype=float)
    pos = np.column_stack([x + offset[0], y + offset[1]])
    vel = np.column_stack([vx, vy])
    acc = None
    if afirst is not None and asecond is not None:
        ax, ay = _apply_frame(np.asarray(afirst, float), np.asarray(asecond, float), frame)
        acc = np.column_stack([ax, ay])
    finite = np.isfinite(t) & np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
    if acc is not None:
        finite &= np.isfinite(acc).all(axis=1)
    t, pos, vel = t[finite], pos[finite], vel[finite]
    acc = acc[finite] if acc is not None else None
    if len(t) < 2:
        return None
    order = np.argsort(t, kind="stable")
    keep = np.ones(len(t), dtype=bool)
    ts = t[order]
    keep[1:] = np.diff(ts) > 1e-9
    idx = order[keep]
    if len(idx) < 2:
        return None
    return Trajectory(
        t=t[idx],
        pos=pos[idx],
        vel=vel[idx],
        acc=acc[idx] if acc is not None else None,
        source=source,
    )


# ---------------------------------------------------------------------------
# PX4 Autopilot ULog
# ---------------------------------------------------------------------------


def _ulog_to_trajectory(
    ulog,
    topic: str,
    frame: str,
    offset: tuple[float, float],
    source: str,
) -> Trajectory | None:
    """Convert a (pyulog-like) ULog object; separated for unit testing."""
    data = ulog.get_dataset(topic).data
    t = np.asarray(data["timestamp"], dtype=float) * 1e-6
    acc_first = data.get("ax")
    acc_second = data.get("ay")
    return _finalize_trajectory(
        t,
        data["x"],
        data["y"],
        data["vx"],
        data["vy"],
        acc_first,
        acc_second,
        frame,
        offset,
        source,
    )


def load_px4_ulog(
    path: str | Path,
    topic: str = "vehicle_local_position",
    frame: str = "ned",
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[Trajectory]:
    """Load one PX4 ``.ulg`` flight log (requires ``pyulog``)."""
    try:
        from pyulog import ULog
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PX4 ULog support requires pyulog; install with `pip install -e '.[flightstack]'`"
        ) from exc
    ulog = ULog(str(path), message_name_filter_list=[topic])
    traj = _ulog_to_trajectory(ulog, topic, frame, offset, str(path))
    return [traj] if traj is not None else []


# ---------------------------------------------------------------------------
# ArduPilot DataFlash / MAVLink telemetry logs
# ---------------------------------------------------------------------------

_ARDUPILOT_AUTO_MESSAGES = ("LOCAL_POSITION_NED", "XKF1", "NKF1")


def _mavlink_records_to_trajectory(
    records: list[tuple[float, float, float, float, float]],
    frame: str,
    offset: tuple[float, float],
    source: str,
) -> Trajectory | None:
    if not records:
        return None
    arr = np.asarray(records, dtype=float)
    return _finalize_trajectory(
        arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], None, None, frame, offset, source
    )


def load_ardupilot_log(
    path: str | Path,
    message: str = "auto",
    frame: str = "ned",
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[Trajectory]:
    """Load one ArduPilot ``.bin``/``.log``/``.tlog`` (requires ``pymavlink``).

    Telemetry logs are read through ``LOCAL_POSITION_NED``; DataFlash logs
    through the EKF position estimate (``XKF1``, falling back to ``NKF1``).
    """
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "ArduPilot log support requires pymavlink; install with `pip install -e '.[flightstack]'`"
        ) from exc
    wanted = _ARDUPILOT_AUTO_MESSAGES if message == "auto" else (message,)
    conn = mavutil.mavlink_connection(str(path))
    per_type: dict[str, list[tuple[float, float, float, float, float]]] = {m: [] for m in wanted}
    while True:
        msg = conn.recv_match(type=list(wanted), blocking=False)
        if msg is None:
            break
        mtype = msg.get_type()
        stamp = float(getattr(msg, "_timestamp", 0.0))
        if mtype == "LOCAL_POSITION_NED":
            per_type[mtype].append((stamp, msg.x, msg.y, msg.vx, msg.vy))
        elif mtype in ("XKF1", "NKF1"):
            # Multi-core EKF logs repeat the state per core; keep core 0.
            core = getattr(msg, "C", 0)
            if core in (0, None):
                per_type[mtype].append((stamp, msg.PN, msg.PE, msg.VN, msg.VE))
    for mtype in wanted:
        traj = _mavlink_records_to_trajectory(per_type[mtype], frame, offset, f"{path}:{mtype}")
        if traj is not None:
            return [traj]
    return []


# ---------------------------------------------------------------------------
# Generic CSV trajectories
# ---------------------------------------------------------------------------


def load_csv_log(
    path: str | Path,
    frame: str = "xy",
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[Trajectory]:
    """Load CSV with columns ``t,x,y,vx,vy[,ax,ay][,episode]``.

    An ``episode`` column splits the file into independent trajectories, which
    is how the SITL recorder stores multiple flights in one file.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    required = {"t", "x", "y", "vx", "vy"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path}: CSV log missing columns {sorted(missing)}")
    has_acc = "ax" in rows[0] and "ay" in rows[0]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("episode", "0")), []).append(row)
    trajectories = []
    for episode, group in groups.items():
        t = np.array([float(r["t"]) for r in group])
        x = np.array([float(r["x"]) for r in group])
        y = np.array([float(r["y"]) for r in group])
        vx = np.array([float(r["vx"]) for r in group])
        vy = np.array([float(r["vy"]) for r in group])
        ax = np.array([float(r["ax"]) for r in group]) if has_acc else None
        ay = np.array([float(r["ay"]) for r in group]) if has_acc else None
        traj = _finalize_trajectory(t, x, y, vx, vy, ax, ay, frame, offset, f"{path}:{episode}")
        if traj is not None:
            trajectories.append(traj)
    return trajectories


# ---------------------------------------------------------------------------
# Dispatch and dataset conversion
# ---------------------------------------------------------------------------

_DEFAULT_FRAMES = {"px4_ulog": "ned", "ardupilot_log": "ned", "csv": "xy"}


def load_trajectories(
    patterns: tuple[str, ...] | list[str],
    source: str,
    frame: str = "auto",
    topic: str = "vehicle_local_position",
    message: str = "auto",
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[Trajectory]:
    """Expand glob patterns and load every matching log for ``source``."""
    if source not in _DEFAULT_FRAMES:
        raise ValueError(f"unknown flight log source: {source!r}")
    resolved_frame = _DEFAULT_FRAMES[source] if frame == "auto" else frame
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob(pattern, recursive=True))
        if not matched and Path(pattern).exists():
            matched = [pattern]
        paths.extend(matched)
    if not paths:
        raise FileNotFoundError(
            f"no flight logs matched patterns: {list(patterns)} "
            f"(searched from {Path.cwd()}). Point data.logs at existing files, or if you "
            "have no real flight logs: (1) generate demo CSV logs with "
            "`python scripts/make_demo_logs.py` and use configs/demo_csv.yaml, "
            "(2) record from a running SITL with `python -m geofence_qnn.cli sitl-record`, "
            "(3) download public .ulg logs from https://logs.px4.io, or "
            "(4) use synthetic data with a flight-stack teacher (configs/smoke_flightstack.yaml)."
        )
    trajectories: list[Trajectory] = []
    for path in paths:
        if source == "px4_ulog":
            trajectories.extend(load_px4_ulog(path, topic, resolved_frame, offset))
        elif source == "ardupilot_log":
            trajectories.extend(load_ardupilot_log(path, message, resolved_frame, offset))
        else:
            trajectories.extend(load_csv_log(path, resolved_frame, offset))
    if not trajectories:
        raise ValueError(f"flight logs matched but contained no usable trajectories: {paths}")
    return trajectories


def resample_trajectory(traj: Trajectory, dt: float) -> Trajectory:
    """Resample onto the control period grid with linear interpolation."""
    duration = float(traj.t[-1] - traj.t[0])
    n = int(duration / dt) + 1
    grid = traj.t[0] + dt * np.arange(n)
    pos = np.column_stack([np.interp(grid, traj.t, traj.pos[:, k]) for k in range(2)])
    vel = np.column_stack([np.interp(grid, traj.t, traj.vel[:, k]) for k in range(2)])
    acc = None
    if traj.acc is not None:
        acc = np.column_stack([np.interp(grid, traj.t, traj.acc[:, k]) for k in range(2)])
    return Trajectory(t=grid, pos=pos, vel=vel, acc=acc, source=traj.source)


def trajectories_to_dataset(
    trajectories: list[Trajectory],
    dt: float,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    amax: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert trajectories to ``(states, features, normalized_actions)``.

    The action at step ``k`` is the logged acceleration when available and
    otherwise ``(v[k+1] - v[k]) / dt``, matching the control semantics of the
    double-integrator plant. States inside the expanded geofence are dropped,
    consistent with the synthetic sampler: the QNN is only trained and
    verified on the region it is allowed to occupy.
    """
    states_list, actions_list = [], []
    for raw in trajectories:
        traj = resample_trajectory(raw, dt)
        if len(traj) < 2:
            continue
        if traj.acc is not None:
            actions = traj.acc[:-1]
        else:
            actions = np.diff(traj.vel, axis=0) / dt
        states = np.column_stack([traj.pos[:-1], traj.vel[:-1]])
        keep = np.array([not geofence.contains(s[:2], margin=margin) for s in states])
        keep &= np.abs(states[:, 2:]).max(axis=1) <= 1.5 * vmax
        if not keep.any():
            continue
        states_list.append(states[keep])
        actions_list.append(np.clip(actions[keep], -amax, amax))
    if not states_list:
        raise ValueError(
            "no usable state-action samples outside the geofence in the flight logs; "
            "check that data.offset places the logged positions in the experiment frame "
            "and that the flights actually move (velocities below 1.5*vmax, positions "
            "outside the expanded forbidden box)"
        )
    states = np.vstack(states_list)
    actions = np.vstack(actions_list)
    x = np.vstack([state_features(s, goal, geofence, position_scale, vmax) for s in states])
    y = actions / amax
    return states, x, y


def make_flight_log_dataset(
    patterns: tuple[str, ...] | list[str],
    source: str,
    n: int,
    seed: int,
    dt: float,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    amax: float,
    margin: float,
    frame: str = "auto",
    topic: str = "vehicle_local_position",
    message: str = "auto",
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full log-to-dataset pipeline with deterministic shuffling/subsampling."""
    trajectories = load_trajectories(patterns, source, frame, topic, message, offset)
    states, x, y = trajectories_to_dataset(
        trajectories, dt, goal, geofence, position_scale, vmax, amax, margin
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    if n and len(order) > n:
        order = order[:n]
    return states[order], x[order], y[order]
