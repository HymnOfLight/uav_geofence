from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def parse_network(text: str) -> list[int]:
    return [int(x) for x in text.lower().split("x")]


def main():
    p = argparse.ArgumentParser(description="Run reproducible seed/network sweeps")
    p.add_argument("--base-config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--networks", nargs="+", required=True)
    p.add_argument("--stage", choices=["all", "train", "e0", "e1", "e2", "mc"], default="all")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for network in args.networks:
        hidden = parse_network(network)
        for seed in args.seeds:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["seed"] = seed
            cfg["training"]["hidden"] = hidden
            run_dir = root / f"net_{network}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "input_config.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "geofence_qnn.cli",
                args.stage,
                "--config",
                str(config_path),
                "--output",
                str(run_dir),
            ]
            print("RUN", " ".join(cmd), flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failures.append({"network": network, "seed": seed, "returncode": result.returncode})
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
    if failures:
        print("Failures:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

