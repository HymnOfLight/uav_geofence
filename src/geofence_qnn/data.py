from __future__ import annotations

import numpy as np

from .controller import batch_teacher_actions
from .features import state_features
from .geometry import ForbiddenBox, box_from_tuple


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
    teacher=None,
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
    actions = batch_teacher_actions(states, goal, geofence, amax, margin, teacher=teacher)
    y = actions / amax
    return states, x, y


def make_dataset_from_config(cfg, teacher=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the training dataset from the configured source.

    ``data.source: synthetic`` keeps the legacy teacher-generated data.
    ``px4_ulog`` / ``ardupilot_log`` / ``csv`` load real flight logs from the
    configured stacks; ``data.synthetic_fraction`` optionally tops the log
    data up with synthetic teacher samples for state-space coverage.
    """
    e, t, d = cfg.environment, cfg.training, cfg.data
    geofence = box_from_tuple(e.forbidden_box)
    goal = np.array(e.goal)
    if d.source == "synthetic":
        return make_dataset(
            t.samples,
            cfg.seed,
            np.array(e.world_min),
            np.array(e.world_max),
            e.vmax,
            e.amax,
            goal,
            geofence,
            e.position_scale,
            e.safety_margin,
            teacher=teacher,
        )

    from .flightstack.logs import make_flight_log_dataset

    n_synth = int(round(np.clip(d.synthetic_fraction, 0.0, 1.0) * t.samples))
    n_logs = t.samples - n_synth
    states, x, y = make_flight_log_dataset(
        d.logs,
        d.source,
        n_logs,
        cfg.seed,
        e.dt,
        goal,
        geofence,
        e.position_scale,
        e.vmax,
        e.amax,
        e.safety_margin,
        frame=d.frame,
        topic=d.topic,
        message=d.message,
        offset=d.offset,
    )
    if n_synth > 0:
        s2, x2, y2 = make_dataset(
            n_synth,
            cfg.seed + 1,
            np.array(e.world_min),
            np.array(e.world_max),
            e.vmax,
            e.amax,
            goal,
            geofence,
            e.position_scale,
            e.safety_margin,
            teacher=teacher,
        )
        states = np.vstack([states, s2])
        x = np.vstack([x, x2])
        y = np.vstack([y, y2])
        order = np.random.default_rng(cfg.seed + 2).permutation(len(x))
        states, x, y = states[order], x[order], y[order]
    return states, x, y
