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
    backend = None
    nccl_available = False
    gloo_available = False
    if torch:
        backend = getattr(torch.distributed, "is_available", lambda: False)()
        nccl_available = getattr(
            torch.distributed, "is_nccl_available", lambda: False
        )()
        gloo_available = getattr(
            torch.distributed, "is_gloo_available", lambda: False
        )()
    return {
        "torch_importable": torch is not None,
        "distributed_available": backend,
        "nccl_available": nccl_available,
        "gloo_available": gloo_available,
        "import_error": IMPORT_ERROR,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = collect()
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
