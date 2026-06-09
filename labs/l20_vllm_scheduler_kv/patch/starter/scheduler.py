"""L21 Patch · vLLM-shaped scheduler and KV cache manager."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class KVBlock:
    block_id: int
    owner_request_id: str | None = None


class KVCacheManager:
    """Small vLLM-v1-shaped KV cache block manager."""

    def __init__(self, num_blocks: int = 16) -> None:
        self.blocks = [KVBlock(block_id) for block_id in range(num_blocks)]

    @property
    def usage(self) -> float:
        used = sum(block.owner_request_id is not None for block in self.blocks)
        return round(used / max(len(self.blocks), 1), 4)

    def allocate_slots(self, request_id: str, num_blocks: int) -> list[int]:
        # TODO(student): allocate the first num_blocks free blocks and mark ownership.
        # Raise RuntimeError("KV cache exhausted") if there are not enough free blocks.
        raise NotImplementedError("L21: implement allocate_slots")

    def free(self, request_id: str) -> None:
        # TODO(student): clear owner_request_id for all blocks owned by request_id.
        raise NotImplementedError("L21: implement free")

    def snapshot(self) -> dict:
        # TODO(student): return total_blocks, usage, free_blocks, used_blocks.
        raise NotImplementedError("L21: implement snapshot")


@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    prompt_token_ids: list[str] = field(default_factory=list)
    output_tokens: list[str] = field(default_factory=list)
    kv_blocks: list[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled_prefill: list[str]
    scheduled_decode: list[str]
    finished: list[str]


class Scheduler:
    """Minimal vLLM-like scheduler: waiting -> running -> decode -> finished."""

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_running_reqs: int = 4) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.max_num_running_reqs = max_num_running_reqs
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.finished: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        request.prompt_token_ids = request.prompt.split()
        self.waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        # TODO(student): admit waiting requests while running capacity and KV blocks allow.
        # TODO(student): finished running requests free their KV blocks.
        # TODO(student): unfinished running requests are scheduled for one decode step.
        raise NotImplementedError("L21: implement Scheduler.schedule")
