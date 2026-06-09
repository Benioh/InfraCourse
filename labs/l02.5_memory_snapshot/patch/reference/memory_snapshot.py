"""Reference solution for L03 Patch · Memory Snapshot."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: Tuple[str, ...]
    timestamp: float


def get_caller_stack(depth: int = 4) -> Tuple[str, ...]:
    frames = inspect.stack()[1 : 1 + depth]
    return tuple(f"{f.filename}:{f.lineno}:{f.function}" for f in frames)


class MemoryTracker:
    def __init__(self) -> None:
        self._events: List[Tuple[str, AllocEvent]] = []
        self._live: Dict[int, AllocEvent] = {}
        self._enabled: bool = False
        self._next_addr: int = 0x10000

    def start(self) -> None:
        self._enabled = True

    def stop(self) -> None:
        self._enabled = False

    def alloc(self, size: int, stack: Tuple[str, ...]) -> int:
        if not self._enabled:
            return -1
        addr = self._next_addr
        self._next_addr += size + 64
        ev = AllocEvent(addr=addr, size=size, stack=tuple(stack), timestamp=time.monotonic())
        self._events.append(("alloc", ev))
        self._live[addr] = ev
        return addr

    def free(self, addr: int) -> None:
        if not self._enabled or addr < 0:
            return
        ev = self._live.pop(addr, None)
        if ev is not None:
            self._events.append(("free", ev))

    def dump_snapshot(self) -> dict:
        return {
            "events": list(self._events),
            "live_allocations": list(self._live.values()),
            "total_leaked_bytes": sum(ev.size for ev in self._live.values()),
        }


def find_top_leaks_by_stack(
    snapshot: dict, k: int = 3
) -> List[Tuple[Tuple[str, ...], int]]:
    by_stack: Dict[Tuple[str, ...], int] = {}
    for ev in snapshot["live_allocations"]:
        by_stack[ev.stack] = by_stack.get(ev.stack, 0) + ev.size
    ranked = sorted(by_stack.items(), key=lambda kv: -kv[1])
    return ranked[:k]
