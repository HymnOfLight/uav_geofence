"""Generate PX4 / ArduPilot SITL setup files from an experiment config.

Reads the experiment geometry (forbidden box, safety margin, goal, speed and
acceleration limits) and produces everything needed to configure a SITL (or a
real vehicle) so its geofence protects the same set the QNN is verified
against:

- ``px4_geofence.plan``       QGroundControl plan: exclusion polygon expanded
                              by the safety margin (PX4 has no fence-margin
                              parameter) plus a takeoff-and-cross mission that
                              forces the vehicle to interact with the fence.
- ``px4_params.txt``          `param set` lines to paste into the pxh shell
                              (geofence action/prediction, speed/accel limits
                              matched to the experiment).
- ``ardupilot_geofence.plan`` Same plan with the raw box: ArduPilot applies
                              FENCE_MARGIN itself, so the margin is set via
                              parameters instead of polygon expansion.
- ``ardupilot_params.parm``   Parameter file for `param load` in MAVProxy
                              (fence, avoidance and WPNAV limits).

World frame convention: x = east, y = north relative to ``--home-lat/lon``
(the default is the PX4 SITL default home in Zurich). This matches how the
log loaders map NED logs, so flights recorded against these fences line up
with the experiment geometry without manual offsets.

Usage:
    python scripts/make_sitl_setup.py --config configs/main.yaml --output fences/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from geofence_qnn.config import load_config  # noqa: E402
from geofence_qnn.flightstack.geo import (  # noqa: E402
    DEFAULT_HOME,
    EARTH_M_PER_DEG,
    box_polygon_latlon as box_polygon,
    xy_to_latlon,
)


def _waypoint(seq: int, command: int, lat: float, lon: float, alt: float) -> dict:
    return {
        "type": "SimpleItem",
        "autoContinue": True,
        "command": command,  # 22 = NAV_TAKEOFF, 16 = NAV_WAYPOINT
        "doJumpId": seq,
        "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        "params": [0, 0, 0, None, lat, lon, alt],
        "AMSLAltAboveTerrain": None,
        "Altitude": alt,
        "AltitudeMode": 1,
    }


def make_plan(
    polygon: list[list[float]],
    home: tuple[float, float],
    mission_xy: list[tuple[float, float]],
    altitude: float,
    firmware_type: int,
) -> dict:
    """Minimal QGroundControl .plan with an exclusion fence and a crossing mission."""
    items = []
    seq = 1
    for i, (x, y) in enumerate(mission_xy):
        lat, lon = xy_to_latlon(x, y, *home)
        items.append(_waypoint(seq, 22 if i == 0 else 16, lat, lon, altitude))
        seq += 1
    return {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "QGroundControl",
        "geoFence": {
            "version": 2,
            "circles": [],
            "polygons": [{"inclusion": False, "polygon": polygon, "version": 1}],
            "breachReturn": None,
        },
        "mission": {
            "version": 2,
            "firmwareType": firmware_type,  # 12 = PX4, 3 = ArduPilot
            "globalPlanAltitudeMode": 1,
            "vehicleType": 2,
            "cruiseSpeed": 15,
            "hoverSpeed": 5,
            "items": items,
            "plannedHomePosition": [home[0], home[1], 488.0],
        },
        "rallyPoints": {"version": 2, "points": []},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SITL geofence/mission/parameter files")
    p.add_argument("--config", default="configs/main.yaml")
    p.add_argument("--home-lat", type=float, default=DEFAULT_HOME[0])
    p.add_argument("--home-lon", type=float, default=DEFAULT_HOME[1])
    p.add_argument("--altitude", type=float, default=10.0, help="mission altitude (m, relative)")
    p.add_argument("--output", default="fences")
    args = p.parse_args()

    cfg = load_config(args.config)
    e = cfg.environment
    home = (args.home_lat, args.home_lon)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Mission: start on the west staging side, cross the fence corridor to the
    # goal. With the fence active the autopilot must avoid/hold, which is
    # exactly the behavior the dataset needs.
    xmin = e.forbidden_box[0]
    start_xy = (xmin - 30.0, 0.0)
    mission_xy = [start_xy, tuple(e.goal)]

    px4_plan = make_plan(
        box_polygon(e.forbidden_box, e.safety_margin, home), home, mission_xy, args.altitude, firmware_type=12
    )
    (out / "px4_geofence.plan").write_text(json.dumps(px4_plan, indent=2), encoding="utf-8")

    ap_plan = make_plan(
        box_polygon(e.forbidden_box, 0.0, home), home, mission_xy, args.altitude, firmware_type=3
    )
    (out / "ardupilot_geofence.plan").write_text(json.dumps(ap_plan, indent=2), encoding="utf-8")

    px4_params = [
        "# paste into the pxh> shell of PX4 SITL (or load via QGC parameters)",
        "param set GF_ACTION 2          # geofence breach action: hold",
        "param set GF_PREDICT 1         # predictive braking before the fence",
        f"param set MPC_XY_VEL_MAX {e.vmax:g}    # match experiment vmax",
        f"param set MPC_ACC_HOR {e.amax:g}       # match experiment amax",
        f"param set MPC_DEC_HOR_MAX {e.amax:g}   # braking decel used by GF_PREDICT",
        "param set COM_OBL_RC_ACT 0",
    ]
    (out / "px4_params.txt").write_text("\n".join(px4_params) + "\n", encoding="utf-8")

    ap_params = [
        "# load in MAVProxy with: param load fences/ardupilot_params.parm",
        "# (one of the WPNAV_*/WP_* pairs will warn as unknown depending on the",
        "#  firmware generation; that is expected and harmless)",
        "FENCE_ENABLE     1",
        "FENCE_TYPE       4          # polygon fences",
        "FENCE_ACTION     1          # RTL/Land on breach (avoidance stops before)",
        f"FENCE_MARGIN     {e.safety_margin:g}",
        "AVOID_ENABLE     7          # use fence + proximity avoidance",
        f"AVOID_ACCEL_MAX  {e.amax:g}",
        "AVOID_BACKUP_SPD 0.75",
        "OA_TYPE          1          # BendyRuler: plan around the fence in AUTO (needs reboot)",
        f"WPNAV_SPEED      {e.vmax * 100:g}      # cm/s, stable releases (<= 4.6)",
        f"WPNAV_ACCEL      {e.amax * 100:g}      # cm/s^2, stable releases (<= 4.6)",
        f"WP_SPD           {e.vmax:g}          # m/s, current master (4.8-dev+)",
        f"WP_ACC           {e.amax:g}          # m/s^2, current master (4.8-dev+)",
    ]
    (out / "ardupilot_params.parm").write_text("\n".join(ap_params) + "\n", encoding="utf-8")

    print(f"wrote SITL setup files to {out}/:")
    for name in ["px4_geofence.plan", "px4_params.txt", "ardupilot_geofence.plan", "ardupilot_params.parm"]:
        print(f"  {out / name}")

    # The vehicle must spawn OUTSIDE the fence (a spawn inside fails pre-arm
    # with "Vehicle breaching Polygon fence"); the mission start is a safe spot.
    spawn_lat, spawn_lon = xy_to_latlon(*start_xy, *home)
    print("\nspawn the SITL outside the fence, at the mission start:")
    print(f"  PX4:       PX4_HOME_LAT={spawn_lat:.7f} PX4_HOME_LON={spawn_lon:.7f} make px4_sitl jmavsim")
    print(f"  ArduPilot: sim_vehicle.py -v ArduCopter --custom-location={spawn_lat:.7f},{spawn_lon:.7f},488,0")
    print(f"             (or arducopter binary: --home {spawn_lat:.7f},{spawn_lon:.7f},488,0)")
    print("\nnext steps (details in EXPERIMENT_STEPS.md section 14.5):")
    print("  PX4:           load px4_geofence.plan in QGC, paste px4_params.txt into pxh>, fly the mission")
    print("  ArduPilot GUI: param load, fence upload via QGC plan, fly the mission in AUTO")
    print("  ArduPilot headless (no QGC):")
    print("    python -m geofence_qnn.cli sitl-fly --config", args.config, "--output runs/sitl --url tcp:127.0.0.1:5760")
    print("  record (parallel):")
    print("    python -m geofence_qnn.cli sitl-record --config", args.config, "--output runs/sitl --url tcp:127.0.0.1:5762 --global-home")


if __name__ == "__main__":
    main()
