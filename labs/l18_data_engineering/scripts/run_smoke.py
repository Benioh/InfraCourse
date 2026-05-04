from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.half_mission_runner import run_half_mission  # noqa: E402


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    print(run_half_mission("l18_data_engineering", args.run_id, args.mode))
