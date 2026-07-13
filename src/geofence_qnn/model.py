from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MLP:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @classmethod
    def create(cls, dims: list[int], seed: int = 0) -> "MLP":
        rng = np.random.default_rng(seed)
        weights, biases = [], []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float64))
            biases.append(np.zeros(fan_out, dtype=np.float64))
        return cls(weights, biases)

    def forward(self, x: np.ndarray, return_cache: bool = False):
        h = np.asarray(x, dtype=np.float64)
        activations = [h]
        preacts = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w.T + b
            preacts.append(z)
            h = np.maximum(z, 0.0) if i < len(self.weights) - 1 else z
            activations.append(h)
        return (h, (activations, preacts)) if return_cache else h

    def predict_action(self, features: np.ndarray, amax: float) -> np.ndarray:
        return np.clip(self.forward(features), -1.0, 1.0) * amax

    def save(self, path: str | Path) -> None:
        payload = {"layers": np.array([len(self.weights)], dtype=np.int64)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"w{i}"] = w
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(path)
        n = int(data["layers"][0])
        return cls([data[f"w{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)])


def train_mlp(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[dict[str, float]]:
    """Small deterministic Adam trainer implemented in NumPy."""
    rng = np.random.default_rng(seed)
    m_w = [np.zeros_like(w) for w in model.weights]
    v_w = [np.zeros_like(w) for w in model.weights]
    m_b = [np.zeros_like(b) for b in model.biases]
    v_b = [np.zeros_like(b) for b in model.biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    history: list[dict[str, float]] = []
    n = len(x)
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = x[idx], y[idx]
            pred, (acts, preacts) = model.forward(xb, return_cache=True)
            diff = pred - yb
            losses.append(float(np.mean(diff * diff)))
            delta = 2.0 * diff / len(xb)
            gw = [np.empty_like(w) for w in model.weights]
            gb = [np.empty_like(b) for b in model.biases]
            for i in reversed(range(len(model.weights))):
                gw[i] = delta.T @ acts[i] + weight_decay * model.weights[i]
                gb[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ model.weights[i]) * (preacts[i - 1] > 0.0)
            step += 1
            for i in range(len(model.weights)):
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * gw[i]
                v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gw[i] * gw[i])
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * gb[i]
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gb[i] * gb[i])
                mw_hat = m_w[i] / (1 - beta1**step)
                vw_hat = v_w[i] / (1 - beta2**step)
                mb_hat = m_b[i] / (1 - beta1**step)
                vb_hat = v_b[i] / (1 - beta2**step)
                model.weights[i] -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
                model.biases[i] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
        pred_all = model.forward(x)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean((pred_all - y) ** 2)),
                "batch_loss": float(np.mean(losses)),
            }
        )
    return history

