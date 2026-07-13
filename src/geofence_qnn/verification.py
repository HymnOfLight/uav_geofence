from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np

from .features import feature_bounds_from_state_cell, state_features
from .geometry import ForbiddenBox
from .quantization import Int8MLP, round_half_away
from .smt import verify_action_halfspace


@dataclass(frozen=True)
class StateCell:
    lo: np.ndarray
    hi: np.ndarray
    face: str

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


def quantized_feature_bounds(flo: np.ndarray, fhi: np.ndarray, qscale: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.clip(round_half_away(flo * qscale), -127, 127).astype(np.int16)
    hi = np.clip(round_half_away(fhi * qscale), -127, 127).astype(np.int16)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def make_boundary_cells(
    n: int,
    rng: np.random.Generator,
    geofence: ForbiddenBox,
    band: float,
    width: float,
    vmax: float,
) -> list[StateCell]:
    cells: list[StateCell] = []
    faces = ["left", "right", "bottom", "top"]
    for i in range(n):
        face = faces[i % 4]
        if face in ("left", "right"):
            x = rng.uniform(geofence.xmin - band, geofence.xmin) if face == "left" else rng.uniform(geofence.xmax, geofence.xmax + band)
            y = rng.uniform(geofence.ymin - band / 2, geofence.ymax + band / 2)
        else:
            x = rng.uniform(geofence.xmin - band / 2, geofence.xmax + band / 2)
            y = rng.uniform(geofence.ymin - band, geofence.ymin) if face == "bottom" else rng.uniform(geofence.ymax, geofence.ymax + band)
        vx, vy = rng.uniform(-0.65 * vmax, 0.65 * vmax, size=2)
        center = np.array([x, y, vx, vy], dtype=float)
        half = np.array([width / 2, width / 2, 0.15, 0.15])
        cells.append(StateCell(center - half, center + half, face))
    return cells


def face_normal(face: str) -> np.ndarray:
    return {
        "left": np.array([-1, 0], dtype=int),
        "right": np.array([1, 0], dtype=int),
        "bottom": np.array([0, -1], dtype=int),
        "top": np.array([0, 1], dtype=int),
    }[face]


def replay_action_property(
    net: Int8MLP,
    cell: StateCell,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    min_outward_q: int,
    samples: int,
    seed: int,
) -> bool:
    rng = np.random.default_rng(seed)
    points = rng.uniform(cell.lo, cell.hi, size=(samples, 4))
    x = np.vstack([state_features(s, goal, geofence, position_scale, vmax) for s in points])
    qout = net.forward_q(net.quantize_input(x)).astype(int)
    vals = qout @ face_normal(cell.face)
    return bool(np.any(vals < min_outward_q))


def run_e1(
    net: Int8MLP,
    cells: list[StateCell],
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    amax: float,
    timeout_ms: int,
    min_outward_accel: float = 0.20,
    replay_samples: int = 128,
    seed: int = 0,
    workers: int = 1,
) -> list[dict]:
    min_q = int(math.ceil((min_outward_accel / amax) * net.qscale))
    jobs = [
        (net, i, cell, goal, geofence, position_scale, vmax, min_q, timeout_ms, replay_samples, seed)
        for i, cell in enumerate(cells)
    ]
    if workers <= 1:
        return [_run_e1_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_e1_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


def _run_e1_job(job) -> dict:
    net, i, cell, goal, geofence, position_scale, vmax, min_q, timeout_ms, replay_samples, seed = job
    flo, fhi = feature_bounds_from_state_cell(cell.lo, cell.hi, goal, geofence, position_scale, vmax)
    qlo, qhi = quantized_feature_bounds(flo, fhi, net.qscale)
    normal = face_normal(cell.face)
    out_lo, out_hi = net.interval_forward_q(qlo, qhi)
    dot_lo = int(sum(c * (out_lo[j] if c >= 0 else out_hi[j]) for j, c in enumerate(normal)))
    dot_hi = int(sum(c * (out_hi[j] if c >= 0 else out_lo[j]) for j, c in enumerate(normal)))
    if dot_lo >= min_q:
        res = {"elapsed_s": 0.0, "solver_status": "not_needed", "status": "SAFE", "method": "interval_prefilter"}
    elif dot_hi < min_q:
        mid = ((qlo.astype(int) + qhi.astype(int)) // 2).astype(np.int16)
        qout = net.forward_q(mid).astype(int)
        res = {
            "elapsed_s": 0.0,
            "solver_status": "not_needed",
            "status": "UNSAFE",
            "method": "interval_prefilter",
            "counterexample_qinput": mid.astype(int).tolist(),
            "counterexample_qoutput": qout.tolist(),
        }
    else:
        res = verify_action_halfspace(
            net,
            qlo,
            qhi,
            normal,
            min_q,
            timeout_ms,
            prefix=f"c{i}",
        )
        res["method"] = "z3_exact"
    res.update(
        {
            "cell_id": i,
            "face": cell.face,
            "state_lo": cell.lo.tolist(),
            "state_hi": cell.hi.tolist(),
            "input_q_lo": qlo.astype(int).tolist(),
            "input_q_hi": qhi.astype(int).tolist(),
            "output_interval_lo": out_lo.astype(int).tolist(),
            "output_interval_hi": out_hi.astype(int).tolist(),
        }
    )
    if res["status"] == "UNSAFE":
        res["replay_found_violation"] = replay_action_property(
            net, cell, goal, geofence, position_scale, vmax, min_q, replay_samples, seed + i
        )
    return res
