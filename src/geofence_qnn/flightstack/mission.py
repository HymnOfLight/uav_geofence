"""Headless ArduPilot SITL mission driver (no ground station needed).

Containers rented for GPU work (AutoDL etc.) usually have no display, so the
QGroundControl steps of the SITL guide are unavailable. This module does the
same work over pure MAVLink: set the fence/avoidance/WPNAV parameters, upload
the exclusion-polygon geofence and the takeoff-and-cross mission derived from
the experiment config, wait for EKF/GPS readiness, arm, take off in GUIDED
and switch to AUTO. Run the recorder in parallel (or read the DataFlash log
afterwards) to collect training data.

Exposed via ``python -m geofence_qnn.cli sitl-fly``. ArduCopter only: PX4
uses different mode/mission semantics; PX4 users load the generated
``.plan`` in QGroundControl instead.
"""

from __future__ import annotations

import time

from .geo import box_polygon_latlon, xy_to_latlon


def _require_mavutil():
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "sitl-fly requires pymavlink; install with `pip install -e '.[flightstack]'`"
        ) from exc
    return mavutil


def ardupilot_param_candidates(vmax: float, amax: float, margin: float) -> list[list[tuple[str, float]]]:
    """Fence/avoidance/navigation parameters matched to the experiment limits.

    Each inner list holds alternatives for different firmware generations:
    stable releases (<= 4.6) use WPNAV_* in cm units, current master
    (4.8-dev+) renamed them to WP_* in SI units. The first existing name wins.
    """
    return [
        [("FENCE_ENABLE", 1)],
        [("FENCE_TYPE", 4)],  # polygon fences
        [("FENCE_ACTION", 1)],  # RTL/Land on breach; avoidance stops before that
        [("FENCE_MARGIN", margin)],
        [("AVOID_ENABLE", 7)],
        [("AVOID_ACCEL_MAX", amax)],
        [("AVOID_BACKUP_SPD", 0.75)],
        [("WPNAV_SPEED", vmax * 100.0), ("WP_SPD", vmax)],  # cm/s vs m/s
        [("WPNAV_ACCEL", amax * 100.0), ("WP_ACC", amax)],  # cm/s^2 vs m/s^2
        # BendyRuler object avoidance: in AUTO the firmware plans around the
        # exclusion fence instead of running into the breach action. Needs a
        # reboot to take effect; fly_mission handles that.
        [("OA_TYPE", 1)],
    ]


def _param_exists(conn, name: str, timeout_s: float = 2.0) -> bool:
    conn.mav.param_request_read_send(conn.target_system, conn.target_component, name.encode("ascii"), -1)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
        if msg is not None and str(msg.param_id) == name:
            return True
    return False


def set_params(conn, candidates: list[list[tuple[str, float]]], timeout_s: float = 10.0) -> None:
    mavutil = _require_mavutil()
    for group in candidates:
        name = value = None
        for cand_name, cand_value in group:
            if _param_exists(conn, cand_name):
                name, value = cand_name, cand_value
                break
        if name is None:
            raise RuntimeError(f"none of {[n for n, _ in group]} exist on this firmware")
        conn.mav.param_set_send(
            conn.target_system,
            conn.target_component,
            name.encode("ascii"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is not None and str(msg.param_id) == name:
                break
        else:
            raise TimeoutError(f"no PARAM_VALUE ack for {name}")
        print(f"  {name} = {value:g}")


def _upload_items(conn, items: list[tuple], mission_type: int, timeout_s: float = 30.0) -> None:
    """Mission-item-protocol upload; items = (frame, command, p1..p4, lat, lon, alt)."""
    mavutil = _require_mavutil()
    conn.mav.mission_count_send(conn.target_system, conn.target_component, len(items), mission_type)
    deadline = time.time() + timeout_s
    sent = set()
    while time.time() < deadline:
        msg = conn.recv_match(
            type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"], blocking=True, timeout=2.0
        )
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ACK":
            if msg.mission_type != mission_type:
                continue
            if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                return
            raise RuntimeError(f"mission upload rejected: MAV_MISSION_RESULT={msg.type}")
        if getattr(msg, "mission_type", mission_type) != mission_type:
            continue
        seq = msg.seq
        frame, command, p1, p2, p3, p4, lat, lon, alt = items[seq]
        conn.mav.mission_item_int_send(
            conn.target_system,
            conn.target_component,
            seq,
            frame,
            command,
            0,  # current
            1,  # autocontinue
            p1, p2, p3, p4,
            int(round(lat * 1e7)),
            int(round(lon * 1e7)),
            alt,
            mission_type,
        )
        sent.add(seq)
    raise TimeoutError(f"mission upload timed out after sending {len(sent)}/{len(items)} items")


def upload_exclusion_fence(conn, polygon: list[list[float]]) -> None:
    """Upload the forbidden box as an exclusion polygon fence."""
    mavutil = _require_mavutil()
    n = len(polygon)
    items = [
        (
            mavutil.mavlink.MAV_FRAME_GLOBAL,
            mavutil.mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION,
            n, 0, 0, 0, lat, lon, 0.0,
        )
        for lat, lon in polygon
    ]
    _upload_items(conn, items, mavutil.mavlink.MAV_MISSION_TYPE_FENCE)


def upload_mission(conn, home: tuple[float, float], waypoints_xy: list[tuple[float, float]], altitude: float) -> int:
    """Upload home + takeoff + waypoints; returns the number of items."""
    mavutil = _require_mavutil()
    rel = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        (mavutil.mavlink.MAV_FRAME_GLOBAL, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, home[0], home[1], 0.0),
        (rel, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, home[0], home[1], altitude),
    ]
    for x, y in waypoints_xy:
        lat, lon = xy_to_latlon(x, y, *home)
        items.append((rel, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, lat, lon, altitude))
    _upload_items(conn, items, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
    return len(items)


def _set_mode(conn, name: str, timeout_s: float = 10.0) -> None:
    mapping = conn.mode_mapping()
    if name not in mapping:
        raise ValueError(f"mode {name} not in {sorted(mapping)}")
    conn.set_mode(mapping[name])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hb = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if hb is not None and conn.flightmode == name:
            return
    raise TimeoutError(f"vehicle did not enter {name}")


def _arm(conn, timeout_s: float) -> None:
    """Arm with retries: pre-arm checks fail until EKF/GPS are ready."""
    mavutil = _require_mavutil()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        end = time.time() + 3.0
        while time.time() < end:
            msg = conn.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=1.0)
            if msg is None:
                continue
            if msg.get_type() == "STATUSTEXT" and "PreArm" in msg.text:
                print(f"  waiting: {msg.text}")
            if conn.motors_armed():
                return
        time.sleep(1.0)
    raise TimeoutError("vehicle did not arm (pre-arm checks kept failing)")


def _connect(url: str, heartbeat_timeout_s: float):
    mavutil = _require_mavutil()
    conn = mavutil.mavlink_connection(url)
    if conn.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
        raise TimeoutError(f"no MAVLink heartbeat from {url} within {heartbeat_timeout_s}s")
    # A bare MAVLink client gets no telemetry until it asks for it (a ground
    # station normally does this); request all streams at 4 Hz.
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
    )
    return conn


def _reboot(conn, url: str, heartbeat_timeout_s: float):
    """Reboot the autopilot (needed for OA_TYPE) and reconnect."""
    mavutil = _require_mavutil()
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    time.sleep(2.0)
    conn.close()
    deadline = time.time() + 60.0
    while True:
        try:
            return _connect(url, heartbeat_timeout_s)
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(2.0)


def fly_mission(
    url: str,
    forbidden_box: tuple[float, float, float, float],
    safety_margin: float,
    goal: tuple[float, float],
    vmax: float,
    amax: float,
    home: tuple[float, float],
    altitude: float = 10.0,
    mission_timeout_s: float = 600.0,
    heartbeat_timeout_s: float = 30.0,
) -> dict:
    """Configure, arm and fly the fence-crossing mission on ArduPilot SITL.

    ``home`` is the coordinate anchor (world origin), not the spawn point:
    the SITL should spawn near the mission start, outside the fence (see the
    spawn location printed by ``scripts/make_sitl_setup.py``).
    """
    mavutil = _require_mavutil()
    conn = _connect(url, heartbeat_timeout_s)
    print(f"connected to system {conn.target_system} on {url}")

    print("setting fence/avoidance/WPNAV/BendyRuler parameters ...")
    set_params(conn, ardupilot_param_candidates(vmax, amax, safety_margin))
    print("rebooting so OA_TYPE (path planner) takes effect ...")
    conn = _reboot(conn, url, heartbeat_timeout_s)

    # ArduPilot applies FENCE_MARGIN itself, so upload the raw box.
    print("uploading exclusion fence polygon ...")
    upload_exclusion_fence(conn, box_polygon_latlon(forbidden_box, 0.0, home))

    start_xy = (forbidden_box[0] - 30.0, 0.0)
    print("uploading takeoff-and-cross mission ...")
    n_items = upload_mission(conn, home, [start_xy, tuple(goal)], altitude)

    print("waiting for GPS 3D fix (can take ~30 s of sim time) ...")
    deadline = time.time() + 180.0
    while time.time() < deadline:
        msg = conn.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2.0)
        if msg is not None and msg.fix_type >= 3:
            break
    else:
        raise TimeoutError("no GPS 3D fix")
    print("waiting for EKF, then arming ...")
    _set_mode(conn, "GUIDED")
    _arm(conn, timeout_s=180.0)
    print("armed; taking off ...")
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude,
    )
    deadline = time.time() + 120.0
    while time.time() < deadline:
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
        if msg is not None and msg.relative_alt >= altitude * 900:  # mm
            break
    else:
        raise TimeoutError("takeoff altitude not reached")
    print(f"airborne at {altitude} m; starting mission (AUTO) ...")
    _set_mode(conn, "AUTO")

    started = time.time()
    reached_last = False
    fence_events = 0
    while time.time() - started < mission_timeout_s:
        msg = conn.recv_match(
            type=["MISSION_ITEM_REACHED", "STATUSTEXT", "HEARTBEAT"], blocking=True, timeout=2.0
        )
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == "MISSION_ITEM_REACHED":
            print(f"  reached mission item {msg.seq}/{n_items - 1}")
            if msg.seq >= n_items - 1:
                reached_last = True
                break
        elif mtype == "STATUSTEXT":
            text = msg.text
            if "fence" in text.lower() or "breach" in text.lower():
                fence_events += 1
                print(f"  fence: {text}")
        elif mtype == "HEARTBEAT" and conn.flightmode in ("RTL", "LAND"):
            # FENCE_ACTION kicked in: the firmware refused to cross.
            print(f"  firmware fence action engaged (mode {conn.flightmode})")
            break
    summary = {
        "url": url,
        "mission_items": n_items,
        "reached_last_waypoint": reached_last,
        "fence_events": fence_events,
        "final_mode": conn.flightmode,
        "elapsed_s": time.time() - started,
    }
    print(f"mission finished: {summary}")
    return summary
