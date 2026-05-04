"""L11.5 Patch · SLiME-shaped versioned rollout manager."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RolloutData:
    rollout_id: int
    prompts: list[str]
    responses: list[str]
    meta_info: dict


class RolloutServer:
    """Tiny SLiME RolloutServer with explicit weight_version."""

    def __init__(self, server_id: str, weight_version: int = 0) -> None:
        self.server_id = server_id
        self.weight_version = weight_version

    def generate(self, prompts: list[str], actor_version: int) -> RolloutData:
        responses = [
            f"response_from_{self.server_id}_w{self.weight_version}_{idx}"
            for idx, _prompt in enumerate(prompts)
        ]
        return RolloutData(
            rollout_id=self.weight_version,
            prompts=prompts,
            responses=responses,
            meta_info={
                "server_id": self.server_id,
                "weight_version": self.weight_version,
                "actor_version": actor_version,
                "staleness": actor_version - self.weight_version,
            },
        )

    def update_weights(self, version: int) -> None:
        self.weight_version = version


class RolloutManager:
    """Tiny RolloutManager that avoids serving from overly stale engines."""

    def __init__(self, servers: list[RolloutServer], max_staleness: int = 1) -> None:
        self.servers = servers
        self.max_staleness = max_staleness
        self.next_server_index = 0

    def generate(self, prompts: list[str], actor_version: int) -> RolloutData:
        # TODO(student): choose the next server whose actor_version - weight_version
        # is <= max_staleness, using round-robin order.
        # Raise RuntimeError if no server is fresh enough.
        raise NotImplementedError("L11.5: implement RolloutManager.generate")

    def update_weights(self, version: int, server_ids: list[str] | None = None) -> list[str]:
        # TODO(student): update all servers, or only server_ids when provided.
        # Return the updated server ids.
        raise NotImplementedError("L11.5: implement RolloutManager.update_weights")

    def freshness(self, actor_version: int) -> dict[str, int]:
        # TODO(student): return {server_id: actor_version - weight_version}.
        raise NotImplementedError("L11.5: implement RolloutManager.freshness")
