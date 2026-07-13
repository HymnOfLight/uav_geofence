"""Behavioral geofence teachers modeled after PX4 Autopilot and ArduPilot.

These classes reproduce the *documented* horizontal geofence avoidance logic of
the two mainstream open-source flight stacks on the 2D double-integrator plant
used throughout this package:

- ``PX4GeofenceTeacher`` follows the PX4 multicopter position-controller
  cascade (``MPC_XY_P`` position P -> velocity setpoint, ``MPC_XY_VEL_P``
  velocity P -> acceleration setpoint) combined with the predictive geofence
  check (``GF_PREDICT``): the current horizontal braking distance is compared
  against the distance to the fence and the vehicle brakes/holds before the
  predicted crossing, pushing back out if the margin is already violated.
- ``ArduPilotFenceTeacher`` follows ArduPilot's ``AC_Avoid`` fence handling:
  the desired velocity component toward the fence is limited by the
  square-root controller ``sqrt(2 * accel * distance_to_margin)`` so the
  vehicle "slides" along the fence, and it backs away at ``AVOID_BACKUP_SPD``
  once inside the margin.

They are faithful behavioral models of the firmware logic, not the firmware
itself; results obtained with them must be reported as such. For
firmware-in-the-loop data use the SITL recorder in
``geofence_qnn.flightstack.sitl`` instead.

Both teachers expose the exact ``action(state, goal, geofence, amax, margin)``
interface of the builtin :class:`geofence_qnn.controller.TeacherController`,
so they can be dropped into data generation, the Monte Carlo comparison and
the runtime shield without touching the verification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ..controller import TeacherController
from ..geometry import ForbiddenBox


def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > limit > 0.0:
        return vec * (limit / norm)
    return vec


def _detour_target(
    p: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    margin: float,
) -> np.ndarray:
    """Deterministic route waypoint around the fence.

    Neither PX4 nor ArduPilot plans paths around a geofence by default; a
    mission normally routes around it and the fence logic is only the safety
    layer. This waypoint plays the role of that mission plan so that
    goal-reaching statistics stay comparable with the builtin teacher. It uses
    the same fixed-side rule as ``TeacherController`` to stay continuous
    enough for a tiny MLP to imitate.
    """
    target = np.asarray(goal, dtype=float).copy()
    route_pad = margin + 7.0
    crossing_left_to_right = p[0] < geofence.xmin and goal[0] > geofence.xmax
    crossing_right_to_left = p[0] > geofence.xmax and goal[0] < geofence.xmin
    in_vertical_shadow = geofence.ymin - route_pad < p[1] < geofence.ymax + route_pad
    if (crossing_left_to_right or crossing_right_to_left) and in_vertical_shadow:
        detour_y = geofence.ymin - route_pad if crossing_left_to_right else geofence.ymax + route_pad
        staging_x = geofence.xmin - route_pad if crossing_left_to_right else geofence.xmax + route_pad
        target = np.array([staging_x, detour_y])
    return target


@dataclass(frozen=True)
class PX4GeofenceTeacher:
    """PX4 Autopilot style position controller with predictive geofence hold.

    Parameter names mirror the PX4 parameters they model:

    - ``mpc_xy_p``: position error to velocity-setpoint P gain (``MPC_XY_P``).
    - ``mpc_xy_vel_p``: velocity error to acceleration P gain (``MPC_XY_VEL_P_ACC``).
    - ``mpc_xy_vel_max``: horizontal speed limit (``MPC_XY_VEL_MAX``).
    - ``mpc_dec_hor_max``: horizontal deceleration used for the braking
      distance prediction (``MPC_DEC_HOR_MAX``); clipped to ``amax`` at runtime.
    - ``gf_predict``: predictive braking before the fence (``GF_PREDICT``).
    - ``reaction_time``: extra look-ahead added to the braking distance,
      modeling controller latency in the prediction.
    - ``use_detour``: enable the mission-layer waypoint around the fence.
    """

    mpc_xy_p: float = 0.95
    mpc_xy_vel_p: float = 1.8
    mpc_xy_vel_max: float = 8.0
    mpc_dec_hor_max: float = 3.0
    gf_predict: bool = True
    reaction_time: float = 0.25
    use_detour: bool = True

    def action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        geofence: ForbiddenBox,
        amax: float,
        margin: float,
    ) -> np.ndarray:
        p = np.asarray(state[:2], dtype=float)
        v = np.asarray(state[2:], dtype=float)
        target = _detour_target(p, goal, geofence, margin) if self.use_detour else np.asarray(goal, float)

        v_sp = _clip_norm(self.mpc_xy_p * (target - p), self.mpc_xy_vel_max)
        u = self.mpc_xy_vel_p * (v_sp - v)

        normal = geofence.nearest_outward_normal(p)
        clearance = geofence.clearance(p) - margin
        decel = min(self.mpc_dec_hor_max, amax)
        v_toward = max(0.0, -float(np.dot(v, normal)))

        if clearance <= 0.0:
            # Margin already violated: the fence action repositions outward
            # while killing the remaining velocity.
            return np.clip(decel * normal - self.mpc_xy_vel_p * v, -amax, amax)

        if self.gf_predict:
            braking_distance = v_toward * v_toward / (2.0 * decel) + v_toward * self.reaction_time
            if v_toward > 0.0 and braking_distance > 0.0 and clearance <= braking_distance:
                # Predicted fence crossing: blend from nominal tracking into a
                # full horizontal brake with an outward bias, mimicking the
                # brake-then-hold fence action while staying continuous.
                severity = float(np.clip(1.0 - clearance / braking_distance, 0.0, 1.0))
                u_brake = -self.mpc_xy_vel_p * v + severity * decel * normal
                u = (1.0 - severity) * u + severity * u_brake

        return np.clip(u, -amax, amax)


@dataclass(frozen=True)
class ArduPilotFenceTeacher:
    """ArduPilot style waypoint controller with AC_Avoid fence velocity limits.

    Parameter names mirror the ArduPilot parameters they model:

    - ``wpnav_speed``: desired horizontal speed toward the target (``WPNAV_SPEED``).
    - ``pos_p``: position error to desired-velocity P gain (``PSC_POSXY_P``).
    - ``vel_p``: velocity error to acceleration P gain (``PSC_VELXY_P``).
    - ``avoid_accel``: acceleration assumed by the square-root speed limiter
      (``AVOID_ACCEL_MAX``); clipped to ``amax`` at runtime.
    - ``avoid_backup_spd``: back-away speed once inside the fence margin
      (``AVOID_BACKUP_SPD``).
    - ``use_detour``: enable the mission-layer waypoint around the fence.

    The fence margin of ``AC_Avoid`` (``FENCE_MARGIN``) is taken from the
    experiment's ``safety_margin`` so all controllers protect the same set.
    """

    wpnav_speed: float = 8.0
    pos_p: float = 1.0
    vel_p: float = 2.0
    avoid_accel: float = 3.0
    avoid_backup_spd: float = 0.75
    use_detour: bool = True

    def action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        geofence: ForbiddenBox,
        amax: float,
        margin: float,
    ) -> np.ndarray:
        p = np.asarray(state[:2], dtype=float)
        v = np.asarray(state[2:], dtype=float)
        target = _detour_target(p, goal, geofence, margin) if self.use_detour else np.asarray(goal, float)

        v_des = _clip_norm(self.pos_p * (target - p), self.wpnav_speed)

        normal = geofence.nearest_outward_normal(p)
        distance = geofence.clearance(p) - margin
        accel = min(self.avoid_accel, amax)

        if distance <= 0.0:
            # Inside the margin: back away at AVOID_BACKUP_SPD.
            v_des = v_des - float(np.dot(v_des, normal)) * normal + self.avoid_backup_spd * normal
        else:
            # AC_Avoid velocity limiting: the component toward the fence may
            # not exceed the sqrt-controller speed for the remaining distance,
            # which lets the vehicle slide along the fence tangentially.
            speed_limit = math.sqrt(2.0 * accel * distance)
            v_toward = -float(np.dot(v_des, normal))
            if v_toward > speed_limit:
                v_des = v_des + (v_toward - speed_limit) * normal

        return np.clip(self.vel_p * (v_des - v), -amax, amax)


def make_teacher(
    backend: str,
    params: dict[str, Any] | None = None,
    vmax: float | None = None,
    amax: float | None = None,
):
    """Instantiate a teacher backend by name (``builtin``/``px4``/``ardupilot``).

    ``vmax``/``amax`` seed sensible defaults for the flight-stack speed and
    deceleration limits unless overridden explicitly in ``params``.
    """
    params = dict(params or {})
    backend = backend.lower()
    if backend == "builtin":
        return TeacherController(**params)
    if backend == "px4":
        if vmax is not None:
            params.setdefault("mpc_xy_vel_max", vmax)
        if amax is not None:
            params.setdefault("mpc_dec_hor_max", 0.75 * amax)
        return PX4GeofenceTeacher(**params)
    if backend == "ardupilot":
        if vmax is not None:
            params.setdefault("wpnav_speed", vmax)
        if amax is not None:
            params.setdefault("avoid_accel", 0.75 * amax)
        return ArduPilotFenceTeacher(**params)
    raise ValueError(f"unknown teacher backend: {backend!r} (expected builtin/px4/ardupilot)")
