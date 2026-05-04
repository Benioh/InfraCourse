from __future__ import annotations

import argparse
import json
from typing import Any


def plan(actor_gpus: int, rollout_gpus: int) -> dict[str, Any]:
    total = max(actor_gpus + rollout_gpus, 1)
    return {
        "actor_gpus": actor_gpus,
        "rollout_gpus": rollout_gpus,
        "rollout_time_sec_estimate": round(10 / max(rollout_gpus, 1), 3),
        "actor_update_time_sec_estimate": round(8 / max(actor_gpus, 1), 3),
        "weight_sync_time_sec_estimate": round(0.5 + total * 0.05, 3),
        "bottleneck": "rollout" if rollout_gpus < actor_gpus else "actor_or_sync",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniInfra actor/rollout resource planner"
    )
    parser.add_argument("--actor-gpus", type=int, default=4)
    parser.add_argument("--rollout-gpus", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            plan(args.actor_gpus, args.rollout_gpus), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
