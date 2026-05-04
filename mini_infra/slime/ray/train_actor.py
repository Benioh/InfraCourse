from __future__ import annotations

from dataclasses import dataclass

from mini_infra.slime.ray.rollout import RolloutData


@dataclass
class ActorTrainResult:
    rollout_id: int
    reward_mean: float
    new_weight_version: int


class TrainRayActor:
    """Tiny SLiME TrainRayActor with the real actor lifecycle names."""

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.weight_version = 0
        self.rollout_manager = None

    def set_rollout_manager(self, rollout_manager) -> None:
        self.rollout_manager = rollout_manager

    def clear_memory(self) -> None:
        return None

    def sleep(self, tags: list[str] | None = None) -> dict:
        return {"sleep": tags or []}

    def wake_up(self, tags: list[str] | None = None) -> dict:
        return {"wake_up": tags or []}

    def train(
        self, rollout_id: int, rollout_data_ref: RolloutData, external_data=None
    ) -> ActorTrainResult:
        reward_mean = 1.0 if rollout_data_ref.responses else 0.0
        self.weight_version += 1
        return ActorTrainResult(
            rollout_id=rollout_id,
            reward_mean=reward_mean,
            new_weight_version=self.weight_version,
        )

    def update_weights(self) -> int:
        if self.rollout_manager is not None:
            self.rollout_manager.update_weights(self.weight_version)
        return self.weight_version

    def save_model(self, rollout_id: int, force_sync: bool = False) -> dict:
        return {
            "rollout_id": rollout_id,
            "force_sync": force_sync,
            "weight_version": self.weight_version,
        }


TrainActor = TrainRayActor
