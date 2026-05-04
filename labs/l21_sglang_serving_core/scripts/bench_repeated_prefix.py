from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/repeated_prefix.yaml")
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "bench_sglang.py"
    command = [
        sys.executable,
        str(script),
        "--config",
        args.config,
        "--workload",
        "repeated_prefix",
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
