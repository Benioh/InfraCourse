from __future__ import annotations

import argparse
import json

from mini_infra.slime.ray.rollout import RolloutManager
from mini_infra.slime.ray.train_actor import TrainRayActor


def train(args) -> dict:
    rollout_manager = RolloutManager()
    actor = TrainRayActor()
    actor.set_rollout_manager(rollout_manager)
    prompts = ["Alice has 3 apples and buys 4 more. Answer:"]
    history = []
    for step in range(args.steps):
        rollout = rollout_manager.generate(prompts, actor_version=actor.weight_version)
        result = actor.train(rollout.rollout_id, rollout)
        actor.update_weights()
        history.append(
            {
                "step": step + 1,
                "rollout_id": rollout.rollout_id,
                "reward_mean": result.reward_mean,
                "weight_version": result.new_weight_version,
                "meta_info": rollout.meta_info,
            }
        )
    return {"history": history, "final_weight_version": actor.weight_version}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini SLiME train loop")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
