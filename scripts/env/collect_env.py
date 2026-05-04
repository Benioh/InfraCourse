from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - diagnostic script
    torch = None


def maybe_run(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def collect() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "env": {
            "PATH": os.environ.get("PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        },
        "torch": {
            "version": getattr(torch, "__version__", None) if torch else None,
            "cuda_available": bool(torch and torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None) if torch else None,
        },
        "commands": {
            "nvidia_smi": maybe_run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]
            ),
            "nvcc": maybe_run(["nvcc", "--version"]),
            "conda": maybe_run(["conda", "--version"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    payload = collect()
    text = json.dumps(payload, indent=2 if args.pretty or args.output else None)
    print(text)
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
