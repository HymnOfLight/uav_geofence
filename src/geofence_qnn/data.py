from __future__ import annotations

import numpy as np

from .controller import batch_teacher_actions
from .features import state_features
from .geometry import ForbiddenBox


def sample_states(
    n: int,
    rng: np.random.Generator,
    world_min: np.ndarray,
    world_max: np.ndarray,
    vmax: float,
    geofence: ForbiddenBox,
    margin: float,
) -> np.ndarray:
    states = []
    while len(states) < n:
        batch = max(1024, n - len(states))
        pos = rng.uniform(world_min, world_max, size=(batch, 2))
        vel = rng.uniform(-0.75 * vmax, 0.75 * vmax, size=(batch, 2))
        for p, v in zip(pos, vel):
            if not geofence.contains(p, margin=margin):
                states.append(np.r_[p, v])
                if len(states) == n:
                    break
    return np.asarray(states)


def make_dataset(
    n: int,
    seed: int,
    world_min: np.ndarray,
    world_max: np.ndarray,
    vmax: float,
    amax: float,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Uniform coverage alone badly under-samples the safety-critical boundary
    # corridor. Use a fixed 40/60 mixture to keep global behavior while making
    # the QNN actually see the detour and braking regimes it will be verified on.
    n_uniform = int(round(0.4 * n))
    uniform = sample_states(n_uniform, rng, world_min, world_max, vmax, geofence, margin)
    focus_min = np.array([max(world_min[0], geofence.xmin - 35.0), max(world_min[1], geofence.ymin - 18.0)])
    focus_max = np.array([min(world_max[0], geofence.xmax + 20.0), min(world_max[1], geofence.ymax + 18.0)])
    focused = sample_states(n - n_uniform, rng, focus_min, focus_max, vmax, geofence, margin)
    states = np.vstack([uniform, focused])
    states = states[rng.permutation(len(states))]
    x = np.vstack([state_features(s, goal, geofence, position_scale, vmax) for s in states])
    actions = batch_teacher_actions(states, goal, geofence, amax, margin)
    y = actions / amax
    return states, x, y
