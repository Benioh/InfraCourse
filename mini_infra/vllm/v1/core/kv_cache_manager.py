from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVBlock:
    block_id: int
    owner_request_id: str | None = None


class KVCacheManager:
    """Small vLLM-v1-shaped KV cache block manager.

    The real vLLM class lives at ``vllm/v1/core/kv_cache_manager.py`` and exposes
    request admission, block allocation, prefix-cache stats, and free/reset APIs.
    MiniInfra keeps the same location and public lifecycle, but reduces blocks to
    integer ids so the scheduler behavior is easy to inspect.
    """

    def __init__(self, num_blocks: int = 16) -> None:
        self.blocks = [KVBlock(block_id) for block_id in range(num_blocks)]

    @property
    def usage(self) -> float:
        used = sum(block.owner_request_id is not None for block in self.blocks)
        return round(used / max(len(self.blocks), 1), 4)

    def allocate_slots(self, request_id: str, num_blocks: int) -> list[int]:
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        free = [block for block in self.blocks if block.owner_request_id is None]
        if len(free) < num_blocks:
            raise RuntimeError("KV cache exhausted")
        allocated = free[:num_blocks]
        for block in allocated:
            block.owner_request_id = request_id
        return [block.block_id for block in allocated]

    def allocate(self, request_id: str, num_blocks: int) -> list[int]:
        return self.allocate_slots(request_id, num_blocks)

    def free(self, request_id: str) -> None:
        for block in self.blocks:
            if block.owner_request_id == request_id:
                block.owner_request_id = None

    def reset_prefix_cache(self) -> bool:
        for block in self.blocks:
            block.owner_request_id = None
        return True

    def snapshot(self) -> dict:
        return {
            "total_blocks": len(self.blocks),
            "usage": self.usage,
            "free_blocks": [
                block.block_id
                for block in self.blocks
                if block.owner_request_id is None
            ],
            "used_blocks": {
                block.block_id: block.owner_request_id
                for block in self.blocks
                if block.owner_request_id is not None
            },
        }
