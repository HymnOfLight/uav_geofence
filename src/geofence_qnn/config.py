from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    dt: float
    integration_dt: float
    vmax: float
    amax: float
    position_scale: float
    world_min: tuple[float, float]
    world_max: tuple[float, float]
    goal: tuple[float, float]
    forbidden_box: tuple[float, float, float, float]
    safety_margin: float


@dataclass(frozen=True)
class TrainingConfig:
    hidden: tuple[int, ...]
    samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    qscale: int


@dataclass(frozen=True)
class VerificationConfig:
    workers: int
    e0_inputs: int
    grid_cells: int
    cell_width: float
    boundary_band: float
    timeout_ms: int
    horizon_steps: int
    max_refinement_depth: int
    initial_box: tuple[float, ...]


@dataclass(frozen=True)
class SimulationConfig:
    episodes: int
    steps: int
    wind_bound: float
    localization_error: float
    shield_horizon: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    environment: EnvironmentConfig
    training: TrainingConfig
    verification: VerificationConfig
    simulation: SimulationConfig


def _tuple(d: dict[str, Any], key: str) -> tuple:
    return tuple(d[key])


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    e = raw["environment"]
    t = raw["training"]
    v = raw["verification"]
    s = raw["simulation"]
    return ExperimentConfig(
        seed=int(raw["seed"]),
        environment=EnvironmentConfig(
            dt=float(e["dt"]),
            integration_dt=float(e["integration_dt"]),
            vmax=float(e["vmax"]),
            amax=float(e["amax"]),
            position_scale=float(e["position_scale"]),
            world_min=_tuple(e, "world_min"),
            world_max=_tuple(e, "world_max"),
            goal=_tuple(e, "goal"),
            forbidden_box=_tuple(e, "forbidden_box"),
            safety_margin=float(e["safety_margin"]),
        ),
        training=TrainingConfig(
            hidden=tuple(int(x) for x in t["hidden"]),
            samples=int(t["samples"]),
            epochs=int(t["epochs"]),
            batch_size=int(t["batch_size"]),
            learning_rate=float(t["learning_rate"]),
            weight_decay=float(t["weight_decay"]),
            qscale=int(t["qscale"]),
        ),
        verification=VerificationConfig(
            workers=int(v.get("workers", 1)),
            e0_inputs=int(v.get("e0_inputs", 1000)),
            grid_cells=int(v["grid_cells"]),
            cell_width=float(v["cell_width"]),
            boundary_band=float(v["boundary_band"]),
            timeout_ms=int(v["timeout_ms"]),
            horizon_steps=int(v["horizon_steps"]),
            max_refinement_depth=int(v["max_refinement_depth"]),
            initial_box=tuple(float(x) for x in v["initial_box"]),
        ),
        simulation=SimulationConfig(
            episodes=int(s["episodes"]),
            steps=int(s["steps"]),
            wind_bound=float(s["wind_bound"]),
            localization_error=float(s["localization_error"]),
            shield_horizon=int(s["shield_horizon"]),
        ),
    )
