"""Reference solution for L30.5 Patch · CUDA Graph + Memory Savor."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch


class GraphCache:
    def __init__(self) -> None:
        self._graphs: Dict[Tuple, Callable] = {}
        self.capture_count: int = 0
        self.replay_count: int = 0

    @staticmethod
    def _shape_key(value) -> Tuple:
        if isinstance(value, torch.Tensor):
            return ("tensor", tuple(value.shape), str(value.dtype))
        return ("scalar", type(value).__name__, value)

    def capture_or_replay(self, fn: Callable, *args, **kwargs) -> Any:
        key = tuple(self._shape_key(a) for a in args) + tuple(
            (k, self._shape_key(v)) for k, v in sorted(kwargs.items())
        )
        if key not in self._graphs:
            self._graphs[key] = fn
            self.capture_count += 1
            return fn(*args, **kwargs)
        self.replay_count += 1
        return self._graphs[key](*args, **kwargs)


class MemorySavor:
    def __init__(self) -> None:
        self._paused: Dict[int, dict] = {}
        self._next_handle: int = 1

    def pause(self, tensor: torch.Tensor) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._paused[handle] = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "data": tensor.detach().clone().cpu(),
        }
        return handle

    def resume(self, handle: int) -> torch.Tensor:
        meta = self._paused.pop(handle)
        return meta["data"].clone()

    def total_paused_bytes(self) -> int:
        return sum(m["data"].numel() * m["data"].element_size() for m in self._paused.values())

    def is_paused(self, handle: int) -> bool:
        return handle in self._paused
