from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Aggregate all_summary.json files from a sweep")
    p.add_argument("--runs", required=True, help="Root directory containing per-run subdirectories")
    p.add_argument("--output", default="aggregate")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()
    root, out = Path(args.runs), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in sorted(root.rglob("all_summary.json")):
        summary = load_json(summary_path)
        cfg_path = summary_path.parent / "resolved_config.yaml"
        config_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        import yaml

        cfg = yaml.safe_load(config_text) if config_text else {}
        e1, e2 = summary["e1"], summary["e2"]
        base = {
            "run": str(summary_path.parent),
            "seed": cfg.get("seed"),
            "hidden": "x".join(map(str, cfg.get("training", {}).get("hidden", []))),
            "e1_cells": e1["cells"],
            "e1_safe": e1["safe"],
            "e1_unsafe_candidates": e1["unsafe_candidates"],
            "e1_unknown": e1["unknown"],
            "e1_replayed_violations": e1["replayed_violations"],
            "e1_median_solver_s": e1["median_solver_s"],
            "e1_p90_solver_s": e1["p90_solver_s"],
            "e2_safe_volume_ratio": e2["safe_volume_ratio"],
            "e2_unsafe_volume_ratio": e2["unsafe_volume_ratio"],
            "e2_unknown_volume_ratio": e2["unknown_volume_ratio"],
            "wall_time_s": summary.get("wall_time_s"),
        }
        mc_by_controller = {r["controller"]: r for r in summary["monte_carlo"]}
        for controller, values in mc_by_controller.items():
            for key in [
                "violation_rate",
                "violation_ci95_lo",
                "violation_ci95_hi",
                "goal_success_rate",
                "min_clearance_min",
                "shield_intervention_rate",
                "shield_decision_ms_p99",
            ]:
                base[f"{controller}_{key}"] = values.get(key)
        rows.append(base)

    if not rows:
        raise SystemExit(f"No all_summary.json files found under {root}")
    keys = sorted({k for row in rows for k in row})
    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    if args.plots:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise SystemExit("Install optional plotting dependencies: pip install -e '.[plots]'") from exc
        labels = [f"{r['hidden']}/s{r['seed']}" for r in rows]
        safe = np.array([r["e1_safe"] / r["e1_cells"] for r in rows])
        unsafe = np.array([r["e1_unsafe_candidates"] / r["e1_cells"] for r in rows])
        unknown = np.array([r["e1_unknown"] / r["e1_cells"] for r in rows])
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.65), 4.8))
        ax.bar(labels, safe, label="SAFE")
        ax.bar(labels, unsafe, bottom=safe, label="UNSAFE candidate")
        ax.bar(labels, unknown, bottom=safe + unsafe, label="UNKNOWN")
        ax.set_ylabel("Cell ratio")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "e1_status_by_run.png", dpi=180)
        plt.close(fig)

        controllers = ["teacher", "float", "int8", "int8_shield"]
        means = [sum(r.get(f"{c}_violation_rate", 0.0) for r in rows) / len(rows) for c in controllers]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(controllers, means)
        ax.set_ylabel("Mean observed violation rate")
        ax.set_ylim(0, max(0.05, max(means) * 1.1))
        fig.tight_layout()
        fig.savefig(out / "monte_carlo_violation_rate.png", dpi=180)
        plt.close(fig)

    print(out / "aggregate.csv")


if __name__ == "__main__":
    main()

