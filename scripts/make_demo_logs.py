"""Generate demo CSV flight logs when no real PX4/ArduPilot logs are available.

Rolls out one of the behavioral flight-stack teachers (px4 / ardupilot /
builtin) in closed loop on the experiment's double-integrator plant and stores
the flights in the CSV trajectory format understood by ``data.source: csv``.
This exercises the exact same log-ingestion pipeline (loading, resampling,
frame handling, action extraction, geofence filtering) as real ULog/DataFlash
files, so the whole log-based experiment path can be validated without any
flight data.

The generated data is still synthetic: report it as "behavioral-model
rollouts", never as real flights. For firmware-level data use SITL recording
(`geofence_qnn.cli sitl-record`) or real logs.

Usage:
    python scripts/make_demo_logs.py --config configs/main.yaml \
        --backend px4 --episodes 40 --output logs/demo
    python -m geofence_qnn.cli all --config configs/demo_csv.yaml --output runs/demo_csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from geofence_qnn.config import load_config
from geofence_qnn.dynamics import step_dynamics
from geofence_qnn.flightstack.teachers import make_teacher
from geofence_qnn.geometry import box_from_tuple


def main() -> None:
    p = argparse.ArgumentParser(description="Generate demo CSV flight logs from a behavioral teacher")
    p.add_argument("--config", default="configs/main.yaml", help="config providing geometry/limits")
    p.add_argument("--backend", default="px4", choices=["builtin", "px4", "ardupilot"])
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--steps", type=int, default=400, help="control periods per episode")
    p.add_argument("--seed", type=int, default=None, help="defaults to the config seed")
    p.add_argument("--output", default="logs/demo", help="directory for the generated CSV log")
    args = p.parse_args()

    cfg = load_config(args.config)
    e = cfg.environment
    geofence = box_from_tuple(e.forbidden_box)
    goal = np.array(e.goal)
    teacher = make_teacher(args.backend, vmax=e.vmax, amax=e.amax)
    rng = np.random.default_rng(cfg.seed if args.seed is None else args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"demo_{args.backend}.csv"

    # Start episodes in the corridor the verification stages care about:
    # upstream of the fence, flying roughly toward the goal behind it.
    x_lo = max(e.world_min[0], geofence.xmin - 45.0)
    x_hi = geofence.xmin - 8.0
    y_pad = min(20.0, abs(e.world_max[1]))
    samples = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "t", "x", "y", "vx", "vy", "ax", "ay"])
        for episode in range(args.episodes):
            state = np.array(
                [
                    rng.uniform(x_lo, x_hi),
                    rng.uniform(-y_pad, y_pad),
                    rng.uniform(0.0, 0.5 * e.vmax),
                    rng.uniform(-0.15 * e.vmax, 0.15 * e.vmax),
                ]
            )
            for k in range(args.steps):
                u = teacher.action(state, goal, geofence, e.amax, e.safety_margin)
                writer.writerow([episode, round(k * e.dt, 6), *state, *u])
                samples += 1
                state = step_dynamics(state, u, e.dt, e.vmax, e.amax)
                if np.linalg.norm(state[:2] - goal) < 2.0 and np.linalg.norm(state[2:]) < 1.0:
                    break

    print(f"wrote {samples} samples over {args.episodes} episodes to {out_csv}")
    print("use it with:")
    print("  data:")
    print("    source: csv")
    print(f"    logs: [\"{out_csv}\"]")


if __name__ == "__main__":
    main()
