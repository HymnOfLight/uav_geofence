from __future__ import annotations

import itertools

import numpy as np

from .geometry import ForbiddenBox


def state_features(
    state: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
) -> np.ndarray:
    """Six affine features, clipped to the training normalization range."""
    p = np.asarray(state[:2], dtype=float)
    v = np.asarray(state[2:], dtype=float)
    raw = np.array(
        [
            (goal[0] - p[0]) / position_scale,
            (goal[1] - p[1]) / position_scale,
            (geofence.center[0] - p[0]) / position_scale,
            (geofence.center[1] - p[1]) / position_scale,
            v[0] / vmax,
            v[1] / vmax,
        ],
        dtype=float,
    )
    return np.clip(raw, -1.0, 1.0)


def feature_interval(
    lo: np.ndarray,
    hi: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval hull for the affine/clipped feature map."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    f_lo = np.array(
        [
            (goal[0] - hi[0]) / position_scale,
            (goal[1] - hi[1]) / position_scale,
            (geofence.center[0] - hi[0]) / position_scale,
            (geofence.center[1] - hi[1]) / position_scale,
            lo[2] / vmax,
            lo[3] / vmax,
        ]
    )
    f_hi = np.array(
        [
            (goal[0] - lo[0]) / position_scale,
            (goal[1] - lo[1]) / position_scale,
            (geofence.center[0] - lo[0]) / position_scale,
            (geofence.center[1] - lo[1]) / position_scale,
            hi[2] / vmax,
            hi[3] / vmax,
        ]
    )
    return np.clip(f_lo, -1.0, 1.0), np.clip(f_hi, -1.0, 1.0)


def feature_bounds_from_state_cell(
    lo: np.ndarray,
    hi: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    return feature_interval(lo, hi, goal, geofence, position_scale, vmax)

