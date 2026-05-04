from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostic script
    torch = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None


def collect() -> dict:
    payload = {
        "torch_importable": torch is not None,
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "cuda_device_count": (
            torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
        ),
        "cuda_version": getattr(torch.version, "cuda", None) if torch else None,
        "devices": [],
        "import_error": IMPORT_ERROR,
    }
    if torch and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            payload["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": ".".join(
                        map(str, torch.cuda.get_device_capability(index))
                    ),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3,
                        2,
                    ),
                }
            )
    return payload


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
