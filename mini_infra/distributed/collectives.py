from __future__ import annotations

import argparse
import json
import os
from typing import Any


def collective_snapshot() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    simulated_sum = sum(range(1, world_size + 1))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "expected_all_reduce_sum": simulated_sum,
        "backend_boundary": (
            "single_process_simulation" if world_size == 1 else "torchrun_env_detected"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniInfra collective semantics snapshot"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = collective_snapshot()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload)


if __name__ == "__main__":
    main()
