from __future__ import annotations

from dataclasses import dataclass, field
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
class DataConfig:
    """Where the state-action training data comes from.

    source:
        synthetic       - teacher controller on sampled states (default, legacy behavior)
        px4_ulog        - PX4 Autopilot .ulg flight logs (requires pyulog)
        ardupilot_log   - ArduPilot DataFlash .bin/.log or MAVLink .tlog (requires pymavlink)
        csv             - generic CSV trajectories with columns t,x,y,vx,vy[,ax,ay]
    """

    source: str = "synthetic"
    logs: tuple[str, ...] = ()
    frame: str = "auto"  # auto | ned | xy
    topic: str = "vehicle_local_position"  # PX4 ULog topic
    message: str = "auto"  # ArduPilot/MAVLink message type override
    offset: tuple[float, float] = (0.0, 0.0)  # shift log positions into the experiment frame
    synthetic_fraction: float = 0.0  # fraction of the dataset drawn from the synthetic teacher


@dataclass(frozen=True)
class TeacherConfig:
    """Which controller acts as the safety teacher / simulation baseline.

    backend:
        builtin    - PID/CBF-style potential-field teacher (legacy behavior)
        px4        - behavioral model of the PX4 Autopilot geofence (GF_PREDICT + hold)
        ardupilot  - behavioral model of the ArduPilot AC_Avoid fence (sqrt-controller slide)
    """

    backend: str = "builtin"
    params: dict[str, Any] = field(default_factory=dict)


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


DEFAULT_CONTROLLERS = ("teacher", "float", "int8", "int8_shield")


@dataclass(frozen=True)
class SimulationConfig:
    episodes: int
    steps: int
    wind_bound: float
    localization_error: float
    shield_horizon: int
    controllers: tuple[str, ...] = DEFAULT_CONTROLLERS


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    environment: EnvironmentConfig
    training: TrainingConfig
    verification: VerificationConfig
    simulation: SimulationConfig
    data: DataConfig = DataConfig()
    teacher: TeacherConfig = TeacherConfig()


def _tuple(d: dict[str, Any], key: str) -> tuple:
    return tuple(d[key])


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    e = raw["environment"]
    t = raw["training"]
    v = raw["verification"]
    s = raw["simulation"]
    d = raw.get("data") or {}
    tc = raw.get("teacher") or {}
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
            controllers=tuple(str(c) for c in s.get("controllers", DEFAULT_CONTROLLERS)),
        ),
        data=DataConfig(
            source=str(d.get("source", "synthetic")),
            logs=tuple(str(p) for p in d.get("logs", [])),
            frame=str(d.get("frame", "auto")),
            topic=str(d.get("topic", "vehicle_local_position")),
            message=str(d.get("message", "auto")),
            offset=tuple(float(x) for x in d.get("offset", (0.0, 0.0))),
            synthetic_fraction=float(d.get("synthetic_fraction", 0.0)),
        ),
        teacher=TeacherConfig(
            backend=str(tc.get("backend", "builtin")),
            params=dict(tc.get("params") or {}),
        ),
    )
