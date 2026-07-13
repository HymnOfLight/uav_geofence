from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamics import interval_step
from .features import feature_interval
from .geometry import ForbiddenBox
from .quantization import Int8MLP, round_half_away


@dataclass
class ReachResult:
    status: str
    depth: int
    lo: np.ndarray
    hi: np.ndarray
    min_clearance_lower_bound: float
    reason: str

    @property
    def volume(self) -> float:
        return float(np.prod(np.maximum(self.hi - self.lo, 0.0)))


def _quant_bounds(lo: np.ndarray, hi: np.ndarray, qscale: int):
    qlo = np.clip(round_half_away(lo * qscale), -127, 127).astype(np.int16)
    qhi = np.clip(round_half_away(hi * qscale), -127, 127).astype(np.int16)
    return np.minimum(qlo, qhi), np.maximum(qlo, qhi)


def verify_box_once(
    net: Int8MLP,
    lo: np.ndarray,
    hi: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    amax: float,
    margin: float,
    dt: float,
    wind_bound: float,
    horizon_steps: int,
    depth: int,
) -> ReachResult:
    initial_lo, initial_hi = lo.copy(), hi.copy()
    min_lb = math.inf
    for step in range(horizon_steps + 1):
        if geofence.interval_inside(lo[:2], hi[:2], margin):
            return ReachResult("UNSAFE", depth, initial_lo, initial_hi, -margin, f"box_inside_forbidden_at_step_{step}")
        if geofence.interval_may_intersect(lo[:2], hi[:2], margin):
            return ReachResult("UNKNOWN", depth, initial_lo, initial_hi, min_lb, f"reachable_box_intersects_at_step_{step}")
        dx = max(geofence.xmin - hi[0], lo[0] - geofence.xmax, 0.0)
        dy = max(geofence.ymin - hi[1], lo[1] - geofence.ymax, 0.0)
        min_lb = min(min_lb, math.hypot(dx, dy) - margin)
        if step == horizon_steps:
            break
        flo, fhi = feature_interval(lo, hi, goal, geofence, position_scale, vmax)
        qlo, qhi = _quant_bounds(flo, fhi, net.qscale)
        ulo_q, uhi_q = net.interval_forward_q(qlo, qhi)
        ulo = np.clip(ulo_q.astype(float) / net.qscale * amax, -amax, amax)
        uhi = np.clip(uhi_q.astype(float) / net.qscale * amax, -amax, amax)
        lo, hi = interval_step(lo, hi, ulo, uhi, dt, vmax, wind_bound)
    return ReachResult("SAFE", depth, initial_lo, initial_hi, min_lb, "no_intersection")


def _split_box(lo: np.ndarray, hi: np.ndarray, position_scale: float, vmax: float):
    normalized_width = (hi - lo) / np.array([position_scale, position_scale, vmax, vmax])
    dim = int(np.argmax(normalized_width))
    mid = (lo[dim] + hi[dim]) / 2
    hi1, lo2 = hi.copy(), lo.copy()
    hi1[dim] = mid
    lo2[dim] = mid
    return (lo.copy(), hi1), (lo2, hi.copy())


def adaptive_reachability(
    net: Int8MLP,
    initial_lo: np.ndarray,
    initial_hi: np.ndarray,
    goal: np.ndarray,
    geofence: ForbiddenBox,
    position_scale: float,
    vmax: float,
    amax: float,
    margin: float,
    dt: float,
    wind_bound: float,
    horizon_steps: int,
    max_depth: int,
) -> list[ReachResult]:
    pending = [(np.asarray(initial_lo, float), np.asarray(initial_hi, float), 0)]
    results: list[ReachResult] = []
    while pending:
        lo, hi, depth = pending.pop()
        res = verify_box_once(
            net, lo, hi, goal, geofence, position_scale, vmax, amax, margin, dt, wind_bound, horizon_steps, depth
        )
        if res.status == "UNKNOWN" and depth < max_depth:
            for child_lo, child_hi in _split_box(lo, hi, position_scale, vmax):
                pending.append((child_lo, child_hi, depth + 1))
        else:
            results.append(res)
    return results
