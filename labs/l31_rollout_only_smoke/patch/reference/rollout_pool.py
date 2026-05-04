"""Reference solution for L10.5 Patch · RolloutPool."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List


class RolloutPool:
    def __init__(
        self,
        generate_fn: Callable[[str], Awaitable[str]],
        max_concurrency: int = 4,
    ) -> None:
        self.generate_fn = generate_fn
        self.max_concurrency = max_concurrency

    async def rollout(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _bounded(p: str) -> str:
            async with sem:
                return await self.generate_fn(p)

        tasks = [_bounded(p) for p in prompts]
        return await asyncio.gather(*tasks)
