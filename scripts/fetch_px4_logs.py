"""Fetch real open-source PX4 flight logs from the public Flight Review database.

Queries the public listing at https://logs.px4.io, filters for quadrotor
flights of a reasonable duration that used a position-controlled mode, and
downloads the original ``.ulg`` files. The result can be used directly with
``configs/px4_public_logs.yaml`` (``data.source: px4_ulog`` +
``data.auto_align: true``):

    python scripts/fetch_px4_logs.py --count 5 --dest logs/px4
    python -m geofence_qnn.cli all --config configs/px4_public_logs.yaml --output runs/px4_public

Each downloaded log is validated by actually parsing it and checking that the
vehicle moved; broken or stationary logs are skipped. Downloaded UUIDs are
real flights uploaded publicly by the PX4 community; cite https://logs.px4.io
as the data source when publishing results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from geofence_qnn.flightstack.logs import load_px4_ulog  # noqa: E402

BASE = "https://logs.px4.io"
_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_duration_s(text: str) -> int:
    m = _DURATION_RE.fullmatch(text.strip())
    if not m or not any(m.groups()):
        return 0
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def list_public_logs(rows: int) -> list[dict]:
    url = (
        f"{BASE}/browse_data_retrieval?draw=1&start=0&length={rows}"
        f"&order%5B0%5D%5Bcolumn%5D=0&order%5B0%5D%5Bdir%5D={quote('desc')}"
    )
    with urlopen(Request(url, headers={"User-Agent": "geofence-qnn/0.1"}), timeout=60) as response:
        payload = json.load(response)
    logs = []
    for row in payload.get("data", []):
        match = re.search(r"log=([0-9a-f-]{36})", row[1])
        if not match:
            continue
        logs.append(
            {
                "uuid": match.group(1),
                "vehicle": row[3],
                "duration_s": parse_duration_s(str(row[7])),
                "modes": str(row[9] or ""),
            }
        )
    return logs


def download(uuid: str, dest: Path) -> Path:
    target = dest / f"{uuid}.ulg"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{BASE}/download?log={uuid}&type=0"
    with urlopen(Request(url, headers={"User-Agent": "geofence-qnn/0.1"}), timeout=300) as response:
        target.write_bytes(response.read())
    return target


def usable(path: Path, min_travel_m: float) -> bool:
    """A log is usable if it parses and the vehicle actually moved."""
    try:
        trajectories = load_px4_ulog(path)
    except Exception as exc:
        print(f"  skipped ({exc})")
        return False
    if not trajectories:
        print("  skipped (no local position data)")
        return False
    span = trajectories[0].pos.max(axis=0) - trajectories[0].pos.min(axis=0)
    if float(max(span)) < min_travel_m:
        print(f"  skipped (moved only {float(max(span)):.1f} m)")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Download real public PX4 flight logs from logs.px4.io")
    p.add_argument("--count", type=int, default=5, help="number of usable logs to download")
    p.add_argument("--dest", default="logs/px4", help="destination directory")
    p.add_argument("--min-duration", type=float, default=120.0, help="minimum flight seconds")
    p.add_argument("--max-duration", type=float, default=1200.0, help="maximum flight seconds")
    p.add_argument("--min-travel", type=float, default=30.0, help="minimum horizontal travel (m)")
    p.add_argument("--mode", default="Position", help="required substring in the flight mode list")
    p.add_argument("--scan", type=int, default=300, help="how many recent public logs to scan")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"scanning up to {args.scan} recent public logs at {BASE} ...")
    candidates = [
        log
        for log in list_public_logs(args.scan)
        if log["vehicle"] == "Quadrotor"
        and args.min_duration <= log["duration_s"] <= args.max_duration
        and args.mode in log["modes"]
    ]
    print(f"{len(candidates)} candidates match (quadrotor, {args.mode} mode, duration filter)")

    kept: list[Path] = []
    for log in candidates:
        if len(kept) >= args.count:
            break
        print(f"downloading {log['uuid']} ({log['duration_s']}s, modes: {log['modes']})")
        try:
            path = download(log["uuid"], dest)
        except Exception as exc:
            print(f"  skipped (download failed: {exc})")
            continue
        if usable(path, args.min_travel):
            kept.append(path)
        else:
            path.unlink(missing_ok=True)

    if not kept:
        raise SystemExit("no usable logs found; relax the filters or retry later")
    print(f"\n{len(kept)} real flight logs ready under {dest}/:")
    for path in kept:
        print(f"  {path}")
    print("\nnext step:")
    print("  python -m geofence_qnn.cli all --config configs/px4_public_logs.yaml --output runs/px4_public")


if __name__ == "__main__":
    main()
