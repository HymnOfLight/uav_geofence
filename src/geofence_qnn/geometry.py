from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ForbiddenBox:
    """Axis-aligned forbidden rectangle [xmin,xmax] x [ymin,ymax]."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> "ForbiddenBox":
        return ForbiddenBox(self.xmin + dx, self.xmax + dx, self.ymin + dy, self.ymax + dy)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2], dtype=float)

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        x, y = float(p[0]), float(p[1])
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )

    def clearance(self, p: np.ndarray) -> float:
        """Signed Euclidean clearance: positive outside, negative inside."""
        x, y = float(p[0]), float(p[1])
        dx = max(self.xmin - x, 0.0, x - self.xmax)
        dy = max(self.ymin - y, 0.0, y - self.ymax)
        if dx > 0.0 or dy > 0.0:
            return math.hypot(dx, dy)
        return -min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)

    def nearest_outward_normal(self, p: np.ndarray) -> np.ndarray:
        """Outward normal of the nearest face, defined also outside the box."""
        x, y = float(p[0]), float(p[1])
        distances = [abs(x - self.xmin), abs(x - self.xmax), abs(y - self.ymin), abs(y - self.ymax)]
        idx = int(np.argmin(distances))
        return (
            np.array([-1.0, 0.0]) if idx == 0 else
            np.array([1.0, 0.0]) if idx == 1 else
            np.array([0.0, -1.0]) if idx == 2 else
            np.array([0.0, 1.0])
        )

    def interval_may_intersect(self, pos_lo: np.ndarray, pos_hi: np.ndarray, margin: float = 0.0) -> bool:
        return not (
            pos_hi[0] < self.xmin - margin
            or pos_lo[0] > self.xmax + margin
            or pos_hi[1] < self.ymin - margin
            or pos_lo[1] > self.ymax + margin
        )

    def interval_inside(self, pos_lo: np.ndarray, pos_hi: np.ndarray, margin: float = 0.0) -> bool:
        return (
            pos_lo[0] >= self.xmin - margin
            and pos_hi[0] <= self.xmax + margin
            and pos_lo[1] >= self.ymin - margin
            and pos_hi[1] <= self.ymax + margin
        )


def box_from_tuple(values: tuple[float, float, float, float]) -> ForbiddenBox:
    return ForbiddenBox(*map(float, values))

