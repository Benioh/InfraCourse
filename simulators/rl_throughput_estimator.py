from __future__ import annotations

import argparse
import json


def estimate(rollout_gpus: int, actor_gpus: int, response_len: int, batch_size: int) -> dict:
    rollout_time = response_len * batch_size / max(rollout_gpus, 1) / 20
    actor_time = batch_size / max(actor_gpus, 1) * 4
    return {
        "rollout_time_sec_estimate": round(rollout_time, 2),
        "actor_update_time_sec_estimate": round(actor_time, 2),
        "bottleneck": "rollout" if rollout_time > actor_time else "actor",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-gpus", type=int, required=True)
    parser.add_argument("--actor-gpus", type=int, required=True)
    parser.add_argument("--response-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.rollout_gpus, args.actor_gpus, args.response_len, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
