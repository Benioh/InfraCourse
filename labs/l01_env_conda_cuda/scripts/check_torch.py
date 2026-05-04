from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostic script
    payload = {"torch_importable": False, "error": str(exc)}
else:
    payload = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
