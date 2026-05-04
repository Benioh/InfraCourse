"""
L10.5 Patch · 异步 Rollout Pool

填空规则：
- TODO(student) 必须自己写
- 允许 asyncio 内置 API
- 不许用 aiomultiprocess / 进程池

完成度自检：
    make patch-test M=l31_rollout_only_smoke
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List


class RolloutPool:
    def __init__(
        self,
        generate_fn: Callable[[str], Awaitable[str]],
        max_concurrency: int = 4,
    ) -> None:
        # TODO(student):
        #   self.generate_fn = generate_fn
        #   self.max_concurrency = max_concurrency
        raise NotImplementedError("L10.5: implement __init__")

    async def rollout(self, prompts: List[str]) -> List[str]:
        """并发跑 generate_fn(prompt)，返回保序的结果列表。"""
        # TODO(student):
        #   if not prompts: return []
        #   sem = asyncio.Semaphore(self.max_concurrency)
        #
        #   async def _bounded(p):
        #       async with sem:
        #           return await self.generate_fn(p)
        #
        #   tasks = [_bounded(p) for p in prompts]
        #   return await asyncio.gather(*tasks)
        raise NotImplementedError("L10.5: implement rollout")
