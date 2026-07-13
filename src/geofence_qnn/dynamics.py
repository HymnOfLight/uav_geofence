from __future__ import annotations

import numpy as np


def step_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    dt: float,
    vmax: float,
    amax: float,
    disturbance: np.ndarray | None = None,
    integration_dt: float | None = None,
) -> np.ndarray:
    """Double-integrator update with optional small integration substeps."""
    x = np.asarray(state, dtype=float).copy()
    u = np.clip(np.asarray(action, dtype=float), -amax, amax)
    w = np.zeros(2) if disturbance is None else np.asarray(disturbance, dtype=float)
    h = integration_dt or dt
    n = max(1, int(round(dt / h)))
    h = dt / n
    for _ in range(n):
        a = u + w
        x[:2] = x[:2] + h * x[2:] + 0.5 * h * h * a
        x[2:] = np.clip(x[2:] + h * a, -vmax, vmax)
    return x


def interval_step(
    lo: np.ndarray,
    hi: np.ndarray,
    action_lo: np.ndarray,
    action_hi: np.ndarray,
    dt: float,
    vmax: float,
    wind_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sound interval step for the linear double-integrator model."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    alo = np.asarray(action_lo, dtype=float) - wind_bound
    ahi = np.asarray(action_hi, dtype=float) + wind_bound
    nlo = np.empty(4, dtype=float)
    nhi = np.empty(4, dtype=float)
    nlo[:2] = lo[:2] + dt * lo[2:] + 0.5 * dt * dt * alo
    nhi[:2] = hi[:2] + dt * hi[2:] + 0.5 * dt * dt * ahi
    nlo[2:] = np.maximum(-vmax, lo[2:] + dt * alo)
    nhi[2:] = np.minimum(vmax, hi[2:] + dt * ahi)
    return nlo, nhi

