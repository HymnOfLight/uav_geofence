from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import MLP


QMIN, QMAX = -127, 127


def round_half_away(x: np.ndarray | float) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))


def round_div_int(n: np.ndarray, d: int) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    return np.where(n >= 0, (n + d // 2) // d, -((-n + d // 2) // d))


@dataclass
class Int8MLP:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    qscale: int = 32

    @classmethod
    def from_float(cls, model: MLP, qscale: int = 32) -> "Int8MLP":
        weights = [np.clip(round_half_away(w * qscale), QMIN, QMAX).astype(np.int8) for w in model.weights]
        biases = [round_half_away(b * qscale * qscale).astype(np.int32) for b in model.biases]
        return cls(weights, biases, qscale)

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        return np.clip(round_half_away(np.asarray(x) * self.qscale), QMIN, QMAX).astype(np.int16)

    def forward_q(self, qx: np.ndarray, return_layers: bool = False):
        h = np.asarray(qx, dtype=np.int64)
        layers = [h.copy()]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            acc = h @ w.astype(np.int64).T + b.astype(np.int64)
            h = round_div_int(acc, self.qscale)
            if i < len(self.weights) - 1:
                h = np.clip(h, 0, QMAX)
            else:
                h = np.clip(h, QMIN, QMAX)
            layers.append(h.copy())
        return (h.astype(np.int16), layers) if return_layers else h.astype(np.int16)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.forward_q(self.quantize_input(x)).astype(float) / self.qscale

    def predict_action(self, features: np.ndarray, amax: float) -> np.ndarray:
        return np.clip(self.forward(features), -1.0, 1.0) * amax

    def interval_forward_q(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l = np.asarray(lo, dtype=np.int64)
        u = np.asarray(hi, dtype=np.int64)
        for i, (w8, b32) in enumerate(zip(self.weights, self.biases)):
            w = w8.astype(np.int64)
            pos = np.maximum(w, 0)
            neg = np.minimum(w, 0)
            acc_lo = l @ pos.T + u @ neg.T + b32
            acc_hi = u @ pos.T + l @ neg.T + b32
            l = round_div_int(acc_lo, self.qscale)
            u = round_div_int(acc_hi, self.qscale)
            if i < len(self.weights) - 1:
                l, u = np.clip(l, 0, QMAX), np.clip(u, 0, QMAX)
            else:
                l, u = np.clip(l, QMIN, QMAX), np.clip(u, QMIN, QMAX)
        return l.astype(np.int16), u.astype(np.int16)

    def save(self, path: str | Path) -> None:
        payload = {"layers": np.array([len(self.weights)]), "qscale": np.array([self.qscale])}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"w{i}"] = w
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Int8MLP":
        data = np.load(path)
        n = int(data["layers"][0])
        return cls([data[f"w{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)], int(data["qscale"][0]))

