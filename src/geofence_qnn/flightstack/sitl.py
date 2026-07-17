"""Record trajectories from a live PX4 / ArduPilot SITL over MAVLink.

This is the firmware-in-the-loop data path: instead of imitating the fence
logic behaviorally, it connects to a running SITL instance (e.g.
``make px4_sitl jmavsim`` or ``sim_vehicle.py -v ArduCopter``), streams the
``LOCAL_POSITION_NED`` estimate and writes the flights into the CSV
trajectory format understood by :mod:`geofence_qnn.flightstack.logs`
(``data.source: csv``). Positions are converted from NED to the experiment's
world frame (x = east, y = north) at write time.

The recorder does not arm the vehicle or change flight modes; missions,
geofence upload and mode changes stay in the operator's hands (QGroundControl,
MAVProxy, mavsdk scripts, ...). Optionally it can stream position setpoints
toward the experiment goal, which works once the vehicle is in OFFBOARD (PX4)
or GUIDED (ArduPilot) mode.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32


def _require_pymavlink():
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "SITL recording requires pymavlink; install with `pip install -e '.[flightstack]'`"
        ) from exc
    return mavutil


MAVLINK_MSG_ID_GLOBAL_POSITION_INT = 33


def record_sitl_trajectories(
    url: str,
    output_csv: str | Path,
    episodes: int = 1,
    duration_s: float = 60.0,
    rate_hz: float = 20.0,
    goal: tuple[float, float] | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
    heartbeat_timeout_s: float = 30.0,
    global_home: tuple[float, float] | None = None,
) -> dict:
    """Record ``episodes`` segments of ``duration_s`` seconds each into CSV.

    Two position sources:

    - ``global_home=(lat, lon)`` (recommended with generated fences): record
      ``GLOBAL_POSITION_INT`` and convert to world x/y around that anchor.
      This is exact regardless of where the EKF origin ended up, so the data
      lines up with fences/missions generated from the same home.
    - ``global_home=None``: record ``LOCAL_POSITION_NED`` relative to the EKF
      origin (usually the spawn point) and apply ``offset``.

    ``goal`` (world x, y) enables streaming of position setpoints toward that
    point at 2 Hz (PX4 OFFBOARD / ArduPilot GUIDED only). Returns a summary
    dict (episodes, samples, duration, output path).
    """
    from .geo import latlon_to_xy

    mavutil = _require_pymavlink()
    conn = mavutil.mavlink_connection(url)
    if conn.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
        raise TimeoutError(f"no MAVLink heartbeat from {url} within {heartbeat_timeout_s}s")
    message_id = (
        MAVLINK_MSG_ID_GLOBAL_POSITION_INT if global_home is not None else MAVLINK_MSG_ID_LOCAL_POSITION_NED
    )
    message_name = "GLOBAL_POSITION_INT" if global_home is not None else "LOCAL_POSITION_NED"
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        int(1e6 / rate_hz),
        0, 0, 0, 0, 0,
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    started = time.time()
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "t", "x", "y", "vx", "vy"])
        for episode in range(episodes):
            episode_start = time.time()
            last_setpoint = 0.0
            while time.time() - episode_start < duration_s:
                now = time.time()
                if goal is not None and now - last_setpoint > 0.5:
                    _send_goal_setpoint(mavutil, conn, goal, offset)
                    last_setpoint = now
                msg = conn.recv_match(type=message_name, blocking=True, timeout=1.0)
                if msg is None:
                    continue
                if global_home is not None:
                    x, y = latlon_to_xy(msg.lat / 1e7, msg.lon / 1e7, *global_home)
                    # GLOBAL_POSITION_INT velocities are NED cm/s.
                    row = [x + offset[0], y + offset[1], msg.vy / 100.0, msg.vx / 100.0]
                else:
                    # NED -> world: x = east + offset_x, y = north + offset_y.
                    row = [msg.y + offset[0], msg.x + offset[1], msg.vy, msg.vx]
                writer.writerow([episode, msg.time_boot_ms / 1e3, *row])
                samples += 1
    return {
        "url": url,
        "episodes": episodes,
        "samples": samples,
        "duration_s": time.time() - started,
        "output": str(output_csv),
    }


def _send_goal_setpoint(mavutil, conn, goal: tuple[float, float], offset: tuple[float, float]) -> None:
    """Stream a position setpoint toward the world-frame goal (NED z is kept)."""
    north = goal[1] - offset[1]
    east = goal[0] - offset[0]
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    )
    conn.mav.set_position_target_local_ned_send(
        0,
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north,
        east,
        0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
    )
