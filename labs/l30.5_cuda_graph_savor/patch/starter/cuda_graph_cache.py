"""
L30.5 Patch · CUDA Graph Cache + Memory Savor (CPU 模拟)

填空规则：
- TODO(student) 必须自己写
- 不许 import torch.cuda.graphs（CPU 不可用）
- 允许 dict / id / tensor.detach().clone()

完成度自检：
    make patch-test M=l30.5_cuda_graph_savor
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch


class GraphCache:
    """模拟 CUDA Graph 的 capture/replay 语义。

    - capture：对某个固定 input shape 第一次见到时记录函数引用
    - replay：相同 shape 再来时直接调用，避免每次重新构造 graph
    """

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
        # 1. 计算 cache key（基于所有 tensor args/kwargs 的 shape）
        # 2. 如果 key 不在 self._graphs：capture（保存 fn 引用），capture_count += 1，eager 跑一次
        # 3. 如果 key 在 self._graphs：replay（call 已 capture 的 fn），replay_count += 1
        # TODO(student):
        #   key = tuple(self._shape_key(a) for a in args) + tuple(
        #       (k, self._shape_key(v)) for k, v in sorted(kwargs.items())
        #   )
        #   if key not in self._graphs:
        #       self._graphs[key] = fn
        #       self.capture_count += 1
        #       return fn(*args, **kwargs)
        #   else:
        #       self.replay_count += 1
        #       return self._graphs[key](*args, **kwargs)
        raise NotImplementedError("L30.5: implement capture_or_replay")


class MemorySavor:
    """模拟 SGLang torch_memory_saver / Megatron CuMemAllocator 的 pause/resume。"""

    def __init__(self) -> None:
        self._paused: Dict[int, dict] = {}
        self._next_handle: int = 1

    def pause(self, tensor: torch.Tensor) -> int:
        """把 tensor 数据 detach+clone 到 CPU 字典；返回 handle。"""
        # TODO(student):
        #   handle = self._next_handle
        #   self._next_handle += 1
        #   self._paused[handle] = {
        #       "shape": tuple(tensor.shape),
        #       "dtype": tensor.dtype,
        #       "data": tensor.detach().clone().cpu(),
        #   }
        #   return handle
        raise NotImplementedError("L30.5: implement pause")

    def resume(self, handle: int) -> torch.Tensor:
        """从字典取出数据，从 pool 移除该 handle，返回 tensor。"""
        # TODO(student):
        #   meta = self._paused.pop(handle)
        #   return meta["data"].clone()
        raise NotImplementedError("L30.5: implement resume")

    def total_paused_bytes(self) -> int:
        """累加所有 paused tensor 的 numel * element_size。"""
        # TODO(student):
        #   return sum(m["data"].numel() * m["data"].element_size() for m in self._paused.values())
        raise NotImplementedError("L30.5: implement total_paused_bytes")

    def is_paused(self, handle: int) -> bool:
        # TODO(student): return handle in self._paused
        raise NotImplementedError("L30.5: implement is_paused")
