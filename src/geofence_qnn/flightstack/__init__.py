"""Integration with mainstream open-source flight stacks (PX4 Autopilot, ArduPilot).

Three complementary entry points:

1. ``teachers``: behavioral models of the documented PX4 / ArduPilot geofence
   avoidance logic that plug into the same ``action(state, goal, geofence,
   amax, margin)`` interface as the builtin teacher. They can generate
   training data and serve as Monte Carlo baselines.
2. ``logs``: loaders that turn real flight logs (PX4 ULog, ArduPilot
   DataFlash/tlog, generic CSV) into the state-action datasets used to train
   and verify the INT8 QNN.
3. ``sitl``: a MAVLink bridge that records trajectories from a live PX4 or
   ArduPilot SITL instance into CSV logs consumable by ``logs``.
"""

from .teachers import ArduPilotFenceTeacher, PX4GeofenceTeacher, make_teacher
from .logs import (
    Trajectory,
    align_trajectories,
    download_log,
    load_ardupilot_log,
    load_csv_log,
    load_px4_ulog,
    load_trajectories,
    make_flight_log_dataset,
    trajectories_to_dataset,
)

__all__ = [
    "ArduPilotFenceTeacher",
    "PX4GeofenceTeacher",
    "make_teacher",
    "Trajectory",
    "align_trajectories",
    "download_log",
    "load_ardupilot_log",
    "load_csv_log",
    "load_px4_ulog",
    "load_trajectories",
    "make_flight_log_dataset",
    "trajectories_to_dataset",
]
