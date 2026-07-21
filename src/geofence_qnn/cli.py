from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .config import load_config
from .data import make_dataset_from_config
from .flightstack.teachers import make_teacher
from .geometry import box_from_tuple
from .io_utils import sha256, write_csv, write_json
from .model import MLP, train_mlp
from .quantization import Int8MLP
from .reachability import adaptive_reachability
from .simulation import run_monte_carlo
from .smt import verify_fixed_input
from .verification import make_boundary_cells, run_e1


def setup(config_path: str, output: str):
    cfg = load_config(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out / "resolved_config.yaml")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": cfg.seed,
    }
    try:
        import z3
        env["z3"] = z3.get_version_string()
    except Exception as exc:
        env["z3"] = f"unavailable: {exc}"
    write_json(out / "environment.json", env)
    return cfg, out


def teacher_from_config(cfg):
    e = cfg.environment
    return make_teacher(cfg.teacher.backend, cfg.teacher.params, vmax=e.vmax, amax=e.amax)


def train_stage(cfg, out: Path):
    e, t = cfg.environment, cfg.training
    states, x, y = make_dataset_from_config(cfg, teacher=teacher_from_config(cfg))
    split = int(0.8 * len(x))
    model = MLP.create([x.shape[1], *t.hidden, 2], cfg.seed)
    history = train_mlp(
        model,
        x[:split],
        y[:split],
        t.epochs,
        t.batch_size,
        t.learning_rate,
        t.weight_decay,
        cfg.seed,
    )
    test_pred = model.forward(x[split:])
    test_mse = float(np.mean((test_pred - y[split:]) ** 2))
    model_path = out / "float_model.npz"
    model.save(model_path)
    qnet = Int8MLP.from_float(model, t.qscale)
    qpath = out / "int8_model.npz"
    qnet.save(qpath)
    q_mse = float(np.mean((qnet.forward(x[split:]) - y[split:]) ** 2))
    write_csv(out / "training_history.csv", history)
    write_json(
        out / "training_summary.json",
        {
            "samples": len(x),
            "train_samples": split,
            "test_samples": len(x) - split,
            "data_source": cfg.data.source,
            "data_logs": list(cfg.data.logs),
            "teacher_backend": cfg.teacher.backend,
            "float_test_mse": test_mse,
            "int8_test_mse": q_mse,
            "float_model_sha256": sha256(model_path),
            "int8_model_sha256": sha256(qpath),
        },
    )
    return model, qnet


def load_models(out: Path):
    return MLP.load(out / "float_model.npz"), Int8MLP.load(out / "int8_model.npz")


def e0_stage(cfg, out: Path, qnet: Int8MLP, samples: int | None = None):
    n = samples or cfg.verification.e0_inputs
    rng = np.random.default_rng(cfg.seed + 100)
    inputs = rng.integers(-127, 128, size=(n, qnet.weights[0].shape[1]), dtype=np.int16)
    rows = [verify_fixed_input(qnet, x, min(cfg.verification.timeout_ms, 5000)) for x in inputs]
    write_csv(out / "e0_consistency.csv", rows)
    summary = {
        "inputs": n,
        "consistent": sum(r["consistent"] for r in rows),
        "inconsistent": sum(not r["consistent"] for r in rows),
        "pass": all(r["consistent"] for r in rows),
    }
    write_json(out / "e0_summary.json", summary)
    if not summary["pass"]:
        raise RuntimeError("E0 failed: SMT and NumPy INT8 semantics disagree")
    return summary


def e1_stage(cfg, out: Path, qnet: Int8MLP):
    e, v = cfg.environment, cfg.verification
    geofence = box_from_tuple(e.forbidden_box)
    cells = make_boundary_cells(v.grid_cells, np.random.default_rng(cfg.seed + 200), geofence, v.boundary_band, v.cell_width, e.vmax)
    rows = run_e1(
        qnet,
        cells,
        np.array(e.goal),
        geofence,
        e.position_scale,
        e.vmax,
        e.amax,
        v.timeout_ms,
        seed=cfg.seed + 300,
        workers=v.workers,
    )
    write_csv(out / "e1_cells.csv", rows)
    summary = {
        "cells": len(rows),
        "safe": sum(r["status"] == "SAFE" for r in rows),
        "unsafe_candidates": sum(r["status"] == "UNSAFE" for r in rows),
        "unknown": sum(r["status"] == "UNKNOWN" for r in rows),
        "replayed_violations": sum(bool(r.get("replay_found_violation")) for r in rows),
        "median_solver_s": float(np.median([r["elapsed_s"] for r in rows])),
        "p90_solver_s": float(np.quantile([r["elapsed_s"] for r in rows], 0.9)),
    }
    write_json(out / "e1_summary.json", summary)
    return summary


def e2_stage(cfg, out: Path, qnet: Int8MLP):
    e, v, s = cfg.environment, cfg.verification, cfg.simulation
    init = np.array(v.initial_box, dtype=float)
    lo = init[[0, 2, 4, 6]]
    hi = init[[1, 3, 5, 7]]
    rows = adaptive_reachability(
        qnet,
        lo,
        hi,
        np.array(e.goal),
        box_from_tuple(e.forbidden_box),
        e.position_scale,
        e.vmax,
        e.amax,
        e.safety_margin,
        e.dt,
        s.wind_bound,
        v.horizon_steps,
        v.max_refinement_depth,
    )
    serial = [
        {
            "status": r.status,
            "depth": r.depth,
            "lo": r.lo.tolist(),
            "hi": r.hi.tolist(),
            "volume": r.volume,
            "min_clearance_lower_bound": r.min_clearance_lower_bound,
            "reason": r.reason,
        }
        for r in rows
    ]
    write_csv(out / "e2_reachability.csv", serial)
    total = sum(r.volume for r in rows)
    summary = {
        "terminal_boxes": len(rows),
        "safe_volume_ratio": sum(r.volume for r in rows if r.status == "SAFE") / total if total else 0.0,
        "unsafe_volume_ratio": sum(r.volume for r in rows if r.status == "UNSAFE") / total if total else 0.0,
        "unknown_volume_ratio": sum(r.volume for r in rows if r.status == "UNKNOWN") / total if total else 0.0,
        "max_depth": max((r.depth for r in rows), default=0),
    }
    write_json(out / "e2_summary.json", summary)
    return summary


def mc_stage(cfg, out: Path, model: MLP, qnet: Int8MLP):
    e, v, s = cfg.environment, cfg.verification, cfg.simulation
    init = np.array(v.initial_box, dtype=float)
    lo, hi = init[[0, 2, 4, 6]], init[[1, 3, 5, 7]]
    summary = run_monte_carlo(
        list(s.controllers),
        s.episodes,
        lo,
        hi,
        np.array(e.goal),
        box_from_tuple(e.forbidden_box),
        e.safety_margin,
        e.dt,
        e.integration_dt,
        e.vmax,
        e.amax,
        e.position_scale,
        s.steps,
        s.wind_bound,
        s.localization_error,
        s.shield_horizon,
        cfg.seed + 400,
        model,
        qnet,
        teacher=teacher_from_config(cfg),
    )
    write_csv(out / "monte_carlo_summary.csv", summary)
    write_json(out / "monte_carlo_summary.json", summary)
    return summary


def run_all(config_path: str, output: str):
    cfg, out = setup(config_path, output)
    started = time.time()
    model, qnet = train_stage(cfg, out)
    summaries = {
        "e0": e0_stage(cfg, out, qnet),
        "e1": e1_stage(cfg, out, qnet),
        "e2": e2_stage(cfg, out, qnet),
        "monte_carlo": mc_stage(cfg, out, model, qnet),
    }
    summaries["wall_time_s"] = time.time() - started
    write_json(out / "all_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def sitl_fly_stage(cfg, out: Path, args):
    from .flightstack.mission import fly_mission

    e = cfg.environment
    summary = fly_mission(
        args.url,
        e.forbidden_box,
        e.safety_margin,
        tuple(e.goal),
        e.vmax,
        e.amax,
        (args.home_lat, args.home_lon),
        altitude=args.altitude,
    )
    write_json(out / "sitl_fly_summary.json", summary)
    return summary


def sitl_record_stage(cfg, out: Path, args):
    from .flightstack.sitl import record_sitl_trajectories

    summary = record_sitl_trajectories(
        args.url,
        out / "sitl_trajectories.csv",
        episodes=args.episodes,
        duration_s=args.duration,
        rate_hz=args.rate,
        goal=tuple(cfg.environment.goal) if args.command_goal else None,
        offset=cfg.data.offset,
        global_home=(args.home_lat, args.home_lon) if args.global_home else None,
    )
    write_json(out / "sitl_record_summary.json", summary)
    return summary


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="UAV geofence INT8 QNN experiments")
    parser.add_argument("command", choices=["all", "train", "e0", "e1", "e2", "mc", "sitl-record", "sitl-fly"])
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", default="runs/smoke")
    sitl = parser.add_argument_group("sitl", "interact with a live PX4/ArduPilot SITL over MAVLink")
    sitl.add_argument("--url", default="udp:127.0.0.1:14550", help="MAVLink connection URL")
    sitl.add_argument("--episodes", type=int, default=1)
    sitl.add_argument("--duration", type=float, default=60.0, help="seconds per episode (sitl-record)")
    sitl.add_argument("--rate", type=float, default=20.0, help="LOCAL_POSITION_NED stream rate (Hz)")
    sitl.add_argument("--command-goal", action="store_true", help="stream position setpoints toward the configured goal")
    from .flightstack.geo import DEFAULT_HOME

    sitl.add_argument("--home-lat", type=float, default=DEFAULT_HOME[0], help="world-frame anchor latitude")
    sitl.add_argument("--home-lon", type=float, default=DEFAULT_HOME[1], help="world-frame anchor longitude")
    sitl.add_argument("--altitude", type=float, default=10.0, help="mission altitude in m (sitl-fly)")
    sitl.add_argument(
        "--global-home",
        action="store_true",
        help="sitl-record: record GLOBAL_POSITION_INT around --home-lat/lon instead of LOCAL_POSITION_NED",
    )
    args = parser.parse_args(argv)
    if args.command == "all":
        return run_all(args.config, args.output)
    cfg, out = setup(args.config, args.output)
    if args.command == "sitl-record":
        print(sitl_record_stage(cfg, out, args))
        return
    if args.command == "sitl-fly":
        print(sitl_fly_stage(cfg, out, args))
        return
    if args.command == "train":
        train_stage(cfg, out)
        return
    model, qnet = load_models(out)
    if args.command == "e0":
        print(e0_stage(cfg, out, qnet))
    elif args.command == "e1":
        print(e1_stage(cfg, out, qnet))
    elif args.command == "e2":
        print(e2_stage(cfg, out, qnet))
    elif args.command == "mc":
        print(mc_stage(cfg, out, model, qnet))


if __name__ == "__main__":
    main()
