from __future__ import annotations

import argparse
import json
from typing import Any


def tensor_parallel_plan(hidden_size: int, tp: int) -> dict[str, Any]:
    return {
        "hidden_size": hidden_size,
        "tp": tp,
        "column_parallel_shard": hidden_size // max(tp, 1),
        "needs_gather_or_reduce": tp > 1,
    }


def pipeline_bubble(microbatches: int, stages: int) -> float:
    return round((stages - 1) / max(microbatches + stages - 1, 1), 4)


def plan(hidden_size: int, tp: int, pp: int, microbatches: int) -> dict[str, Any]:
    return {
        "tp": tensor_parallel_plan(hidden_size, tp),
        "pp": {
            "stages": pp,
            "microbatches": microbatches,
            "bubble_ratio": pipeline_bubble(microbatches, pp),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra TP/PP toy planner")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            plan(args.hidden_size, args.tp, args.pp, args.microbatches),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
