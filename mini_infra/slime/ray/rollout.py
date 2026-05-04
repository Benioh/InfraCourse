from __future__ import annotations

from dataclasses import asdict, dataclass

import argparse
import json

from mini_infra.sglang.srt.managers.scheduler import Req, Scheduler


@dataclass
class RolloutData:
    rollout_id: int
    prompts: list[str]
    responses: list[str]
    meta_info: dict


class RolloutServer:
    """Tiny SLiME RolloutServer backed by the Mini SGLang scheduler."""

    def __init__(self, server_id: str = "rollout-0", weight_version: int = 0) -> None:
        self.server_id = server_id
        self.scheduler = Scheduler()
        self.weight_version = weight_version

    def generate(
        self, prompts: list[str], actor_version: int | None = None
    ) -> RolloutData:
        actor_version = self.weight_version if actor_version is None else actor_version
        responses = []
        for index, prompt in enumerate(prompts):
            request_id = f"{self.server_id}-{self.weight_version}-{index}"
            self.scheduler.add_request(Req(request_id, prompt.split(), []))
        batch = self.scheduler.run_batch()
        for request_id in batch["decode"]:
            responses.append(f"response_from_{request_id}_w{self.weight_version}")
        return RolloutData(
            rollout_id=self.weight_version,
            prompts=prompts,
            responses=responses,
            meta_info={
                "cache_hits": batch["cache_hits"],
                "server_id": self.server_id,
                "weight_version": self.weight_version,
                "actor_version": actor_version,
                "staleness": actor_version - self.weight_version,
            },
        )

    def update_weights(self, version: int) -> None:
        self.weight_version = version


class RolloutManager:
    """Tiny RolloutManager with explicit freshness-aware generate/update methods."""

    def __init__(
        self, servers: list[RolloutServer] | None = None, max_staleness: int = 1
    ) -> None:
        if servers is not None and not servers:
            raise ValueError("at least one rollout server is required")
        self.servers = servers or [RolloutServer()]
        self.max_staleness = max_staleness
        self.next_server_index = 0

    def _fresh_enough(self, server: RolloutServer, actor_version: int) -> bool:
        return actor_version - server.weight_version <= self.max_staleness

    def generate(
        self, prompts: list[str], actor_version: int | None = None
    ) -> RolloutData:
        if actor_version is None:
            actor_version = max(server.weight_version for server in self.servers)
        count = len(self.servers)
        for offset in range(count):
            index = (self.next_server_index + offset) % count
            server = self.servers[index]
            if self._fresh_enough(server, actor_version):
                self.next_server_index = (index + 1) % count
                return server.generate(prompts, actor_version)
        raise RuntimeError("no rollout server is fresh enough")

    def update_weights(
        self, version: int, server_ids: list[str] | None = None
    ) -> list[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini SLiME rollout manager smoke")
    parser.add_argument("--prompt", action="append", default=None)
    args = parser.parse_args()
    prompts = args.prompt or ["Alice has 3 apples and buys 4 more. Answer:"]
    rollout = RolloutManager().generate(prompts)
    print(json.dumps(asdict(rollout), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
