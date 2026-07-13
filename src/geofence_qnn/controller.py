from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ForbiddenBox


@dataclass(frozen=True)
class TeacherController:
    goal_gain: float = 0.16
    velocity_gain: float = 0.75
    barrier_gain: float = 8.0
    influence_distance: float = 18.0

    def action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        geofence: ForbiddenBox,
        amax: float,
        margin: float,
    ) -> np.ndarray:
        p, v = state[:2], state[2:]
        target = np.asarray(goal, dtype=float).copy()
        # A deterministic waypoint prevents the potential-field teacher from
        # aiming through the forbidden rectangle when the goal is opposite it.
        route_pad = margin + 7.0
        crossing_left_to_right = p[0] < geofence.xmin and goal[0] > geofence.xmax
        crossing_right_to_left = p[0] > geofence.xmax and goal[0] < geofence.xmin
        in_vertical_shadow = geofence.ymin - route_pad < p[1] < geofence.ymax + route_pad
        if (crossing_left_to_right or crossing_right_to_left) and in_vertical_shadow:
            # A fixed side keeps the teacher continuous enough for a tiny MLP
            # to imitate. Choosing top/bottom from sign(y) creates a sharp
            # discontinuity near y=0 and makes the smoke model stall.
            detour_y = geofence.ymin - route_pad if crossing_left_to_right else geofence.ymax + route_pad
            staging_x = geofence.xmin - route_pad if crossing_left_to_right else geofence.xmax + route_pad
            target = np.array([staging_x, detour_y])
        u = self.goal_gain * (target - p) - self.velocity_gain * v
        clearance = geofence.clearance(p) - margin
        if clearance < self.influence_distance:
            normal = geofence.nearest_outward_normal(p)
            proximity = np.clip((self.influence_distance - clearance) / self.influence_distance, 0.0, 2.0)
            toward_speed = min(0.0, float(np.dot(v, normal)))
            u += self.barrier_gain * proximity * normal - 1.2 * toward_speed * normal
        return np.clip(u, -amax, amax)


def batch_teacher_actions(
    states: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    amax: float,
    margin: float,
    teacher: TeacherController | None = None,
) -> np.ndarray:
    teacher = teacher or TeacherController()
    return np.vstack([teacher.action(s, goal, geofence, amax, margin) for s in states])
