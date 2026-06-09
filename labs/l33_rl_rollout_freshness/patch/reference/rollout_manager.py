"""Reference solution for L38 Patch."""

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
        if not servers:
            raise ValueError("at least one rollout server is required")
        self.servers = servers
        self.max_staleness = max_staleness
        self.next_server_index = 0

    def _fresh_enough(self, server: RolloutServer, actor_version: int) -> bool:
        return actor_version - server.weight_version <= self.max_staleness

    def generate(self, prompts: list[str], actor_version: int) -> RolloutData:
        count = len(self.servers)
        for offset in range(count):
            index = (self.next_server_index + offset) % count
            server = self.servers[index]
            if self._fresh_enough(server, actor_version):
                self.next_server_index = (index + 1) % count
                return server.generate(prompts, actor_version)
        raise RuntimeError("no rollout server is fresh enough")

    def update_weights(self, version: int, server_ids: list[str] | None = None) -> list[str]:
        allowed = set(server_ids) if server_ids is not None else None
        updated: list[str] = []
        for server in self.servers:
            if allowed is None or server.server_id in allowed:
                server.update_weights(version)
                updated.append(server.server_id)
        return updated

    def freshness(self, actor_version: int) -> dict[str, int]:
        return {
            server.server_id: actor_version - server.weight_version
            for server in self.servers
        }
