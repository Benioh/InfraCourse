"""
L03 Patch · Memory Snapshot · 按 Stack 定位泄露 (CPU 模拟)

填空规则：
- TODO(student) 必须自己写
- 不许 import torch.cuda.memory._record_memory_history
- 允许 inspect.stack / dataclasses / time

完成度自检：
    make patch-test M=l02.5_memory_snapshot
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: Tuple[str, ...]
    timestamp: float


def get_caller_stack(depth: int = 4) -> Tuple[str, ...]:
    """返回 caller 上溯 depth 层的 stack，每帧格式 'file:line:func'。

    跳过本函数自身（depth=1 时拿调用者，不拿自己）。
    """
    # TODO(student):
    #   frames = inspect.stack()[1 : 1 + depth]
    #   return tuple(f"{f.filename}:{f.lineno}:{f.function}" for f in frames)
    raise NotImplementedError("L03: implement get_caller_stack")


class MemoryTracker:
    """记录 alloc/free 的 lightweight tracker，用于本地泄露排查。"""

    def __init__(self) -> None:
        self._events: List[Tuple[str, AllocEvent]] = []
        self._live: Dict[int, AllocEvent] = {}
        self._enabled: bool = False
        self._next_addr: int = 0x10000

    def start(self) -> None:
        # TODO(student): self._enabled = True
        raise NotImplementedError("L03: implement start")

    def stop(self) -> None:
        # TODO(student): self._enabled = False
        raise NotImplementedError("L03: implement stop")

    def alloc(self, size: int, stack: Tuple[str, ...]) -> int:
        """返回新 addr；disabled 时返回 -1（且不改 state）。

        每次分配 addr += size + 64 padding，保证 addr 严格单调递增。
        """
        # TODO(student):
        #   if not self._enabled:
        #       return -1
        #   addr = self._next_addr
        #   self._next_addr += size + 64
        #   ev = AllocEvent(addr=addr, size=size, stack=tuple(stack), timestamp=time.monotonic())
        #   self._events.append(("alloc", ev))
        #   self._live[addr] = ev
        #   return addr
        raise NotImplementedError("L03: implement alloc")

    def free(self, addr: int) -> None:
        """从 _live 中移除 addr 对应的 AllocEvent；不存在则安静返回。"""
        # TODO(student):
        #   if not self._enabled or addr < 0:
        #       return
        #   ev = self._live.pop(addr, None)
        #   if ev is not None:
        #       self._events.append(("free", ev))
        raise NotImplementedError("L03: implement free")

    def dump_snapshot(self) -> dict:
        """返回 {"events", "live_allocations", "total_leaked_bytes"}."""
        # TODO(student):
        #   return {
        #       "events": list(self._events),
        #       "live_allocations": list(self._live.values()),
        #       "total_leaked_bytes": sum(ev.size for ev in self._live.values()),
        #   }
        raise NotImplementedError("L03: implement dump_snapshot")


def find_top_leaks_by_stack(snapshot: dict, k: int = 3) -> List[Tuple[Tuple[str, ...], int]]:
    """按 stack 聚合 leaked allocations 的总 size，返回 [(stack, total_bytes), ...] desc."""
    # TODO(student):
    #   by_stack: dict = {}
    #   for ev in snapshot["live_allocations"]:
    #       by_stack[ev.stack] = by_stack.get(ev.stack, 0) + ev.size
    #   ranked = sorted(by_stack.items(), key=lambda kv: -kv[1])
    #   return ranked[:k]
    raise NotImplementedError("L03: implement find_top_leaks_by_stack")
