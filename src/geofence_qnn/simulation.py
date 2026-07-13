from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .controller import TeacherController
from .dynamics import step_dynamics
from .features import state_features
from .geometry import ForbiddenBox
from .model import MLP
from .quantization import Int8MLP


@dataclass
class EpisodeResult:
    violated: bool
    reached_goal: bool
    min_clearance: float
    shield_interventions: int
    steps: int
    shield_decision_times_ms: list[float]


def controller_action(
    kind: str,
    state_est: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    amax: float,
    margin: float,
    position_scale: float,
    vmax: float,
    float_model: MLP | None,
    int8_model: Int8MLP | None,
    teachers: dict[str, object] | None = None,
) -> np.ndarray:
    # Teacher-style baselines (builtin teacher, PX4/ArduPilot behavioral
    # models, ...) all share the same action(...) interface.
    if teachers and kind in teachers:
        return teachers[kind].action(state_est, goal, geofence, amax, margin)
    if kind == "teacher":
        return TeacherController().action(state_est, goal, geofence, amax, margin)
    feat = state_features(state_est, goal, geofence, position_scale, vmax)
    if kind == "float":
        assert float_model is not None
        return float_model.predict_action(feat, amax)
    if kind in ("int8", "int8_shield"):
        assert int8_model is not None
        return int8_model.predict_action(feat, amax)
    raise ValueError(f"unknown controller: {kind}")


def would_be_unsafe(
    state: np.ndarray,
    action: np.ndarray,
    geofence: ForbiddenBox,
    margin: float,
    dt: float,
    vmax: float,
    amax: float,
    horizon: int,
) -> bool:
    x = state.copy()
    for _ in range(horizon):
        x = step_dynamics(x, action, dt, vmax, amax)
        if geofence.contains(x[:2], margin):
            return True
    return False


def run_episode(
    kind: str,
    initial_state: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    margin: float,
    dt: float,
    integration_dt: float,
    vmax: float,
    amax: float,
    position_scale: float,
    steps: int,
    wind_bound: float,
    localization_error: float,
    shield_horizon: int,
    rng: np.random.Generator,
    float_model: MLP | None = None,
    int8_model: Int8MLP | None = None,
    teachers: dict[str, object] | None = None,
) -> EpisodeResult:
    x = np.asarray(initial_state, float).copy()
    min_clearance = geofence.clearance(x[:2])
    interventions = 0
    violated = geofence.contains(x[:2], margin)
    reached = False
    shield_times: list[float] = []
    for k in range(steps):
        est = x.copy()
        est[:2] += rng.uniform(-localization_error, localization_error, size=2)
        u = controller_action(
            kind, est, goal, geofence, amax, margin, position_scale, vmax, float_model, int8_model, teachers
        )
        if kind == "int8_shield":
            t0 = time.perf_counter_ns()
            unsafe = would_be_unsafe(x, u, geofence, margin, dt, vmax, amax, shield_horizon)
            shield_times.append((time.perf_counter_ns() - t0) / 1e6)
            if unsafe:
                normal = geofence.nearest_outward_normal(x[:2])
                u = np.clip(amax * normal - 1.25 * x[2:], -amax, amax)
                interventions += 1
        wind = rng.uniform(-wind_bound, wind_bound, size=2)
        x = step_dynamics(x, u, dt, vmax, amax, wind, integration_dt)
        min_clearance = min(min_clearance, geofence.clearance(x[:2]))
        if geofence.contains(x[:2], margin):
            violated = True
            break
        if np.linalg.norm(x[:2] - goal) < 2.0 and np.linalg.norm(x[2:]) < 1.0:
            reached = True
            break
    return EpisodeResult(violated, reached, min_clearance, interventions, k + 1, shield_times)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return lo, hi


def run_monte_carlo(
    kinds: list[str],
    episodes: int,
    initial_lo: np.ndarray,
    initial_hi: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    margin: float,
    dt: float,
    integration_dt: float,
    vmax: float,
    amax: float,
    position_scale: float,
    steps: int,
    wind_bound: float,
    localization_error: float,
    shield_horizon: int,
    seed: int,
    float_model: MLP,
    int8_model: Int8MLP,
    teacher: object | None = None,
) -> list[dict]:
    master = np.random.default_rng(seed)
    initial_states = master.uniform(initial_lo, initial_hi, size=(episodes, 4))
    episode_seeds = master.integers(0, 2**32 - 1, size=episodes, dtype=np.uint64)
    teachers: dict[str, object] = {"teacher": teacher or TeacherController()}
    for kind in kinds:
        if kind in ("px4", "ardupilot") and kind not in teachers:
            from .flightstack.teachers import make_teacher

            teachers[kind] = make_teacher(kind, vmax=vmax, amax=amax)
    summaries = []
    for kind in kinds:
        rows = []
        for i in range(episodes):
            rng = np.random.default_rng(int(episode_seeds[i]))
            rows.append(
                run_episode(
                    kind,
                    initial_states[i],
                    goal,
                    geofence,
                    margin,
                    dt,
                    integration_dt,
                    vmax,
                    amax,
                    position_scale,
                    steps,
                    wind_bound,
                    localization_error,
                    shield_horizon,
                    rng,
                    float_model,
                    int8_model,
                    teachers,
                )
            )
        violations = sum(x.violated for x in rows)
        reached = sum(x.reached_goal for x in rows)
        lo_ci, hi_ci = wilson_interval(violations, episodes)
        shield_times = [t for row in rows for t in row.shield_decision_times_ms]
        summaries.append(
            {
                "controller": kind,
                "episodes": episodes,
                "violations": violations,
                "violation_rate": violations / episodes,
                "violation_ci95_lo": lo_ci,
                "violation_ci95_hi": hi_ci,
                "goal_success_rate": reached / episodes,
                "min_clearance_mean": float(np.mean([x.min_clearance for x in rows])),
                "min_clearance_min": float(np.min([x.min_clearance for x in rows])),
                "shield_intervention_rate": float(np.mean([x.shield_interventions > 0 for x in rows])),
                "shield_interventions_mean": float(np.mean([x.shield_interventions for x in rows])),
                "shield_decision_ms_mean": float(np.mean(shield_times)) if shield_times else 0.0,
                "shield_decision_ms_p99": float(np.quantile(shield_times, 0.99)) if shield_times else 0.0,
            }
        )
    return summaries
