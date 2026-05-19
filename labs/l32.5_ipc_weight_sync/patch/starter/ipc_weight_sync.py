"""
L32.5 Patch · CUDA IPC-shaped Weight Sync (CPU simulation)

填空规则：
- TODO(student) 必须自己写
- 不许把 tensor data 塞进序列化的 bytes（违背 IPC 共享语义）
- 允许 pickle / uuid / dataclasses

完成度自检：
    make patch-test M=l32.5_ipc_weight_sync
"""

from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ============================================================================
# 不要修改：IPCStoragePool（GPU IPC 存储的 CPU 模拟）
# ============================================================================
class IPCStoragePool:
    """模拟 CUDA IPC 存储池：
    - put(tensor) → 注册 tensor 拿到 uuid handle，不复制
    - get(handle)  → 按 handle 拿回 tensor（与原 tensor 共享 storage）
    - free(handle) → 释放
    """

    def __init__(self) -> None:
        self._tensors: Dict[str, torch.Tensor] = {}

    def put(self, tensor: torch.Tensor) -> str:
        h = uuid.uuid4().hex
        self._tensors[h] = tensor
        return h

    def get(self, handle: str) -> torch.Tensor:
        if handle not in self._tensors:
            raise KeyError(f"handle {handle} not in pool")
        return self._tensors[handle]

    def free(self, handle: str) -> None:
        self._tensors.pop(handle, None)

    def clear(self) -> None:
        self._tensors.clear()

    def size(self) -> int:
        return len(self._tensors)


# ============================================================================
# 你要实现的部分从这里开始
# ============================================================================

def serialize_handle(tensor: torch.Tensor, pool: IPCStoragePool) -> bytes:
    """模拟 MultiprocessingSerializer.serialize：

    1. 把 tensor 注册进 pool 拿到 handle（uuid 字符串）
    2. 把 (handle, shape, dtype, stride, device) 这种 meta dict pickle 成 bytes
    3. 返回 bytes（不能包含 tensor data 本身）

    返回 bytes 的长度应远小于 tensor 的字节数（numel * element_size）。
    """
    # TODO(student):
    #   handle = pool.put(tensor)
    #   meta = {
    #       "handle": handle,
    #       "shape": list(tensor.shape),
    #       "dtype": str(tensor.dtype),
    #       "stride": list(tensor.stride()),
    #       "device": str(tensor.device),
    #   }
    #   return pickle.dumps(meta)
    raise NotImplementedError("L32.5: implement serialize_handle")


def deserialize_handle(blob: bytes, pool: IPCStoragePool) -> torch.Tensor:
    """模拟 SGLang 侧的 reduce_tensor 反序列化：

    1. pickle.loads 得到 meta
    2. 用 meta["handle"] 从 pool 取出 tensor（共享 storage，不复制）
    3. 返回该 tensor
    """
    # TODO(student):
    #   meta = pickle.loads(blob)
    #   return pool.get(meta["handle"])
    raise NotImplementedError("L32.5: implement deserialize_handle")


@dataclass
class LocalSerializedTensor:
    """对应 verl/sglang 中的 LocalSerializedTensor：每个 source rank 一个 handle blob。"""

    values: List[bytes]

    def get(self, rank: int, pool: IPCStoragePool) -> torch.Tensor:
        # TODO(student): 反序列化 self.values[rank]，从 pool 取出 tensor
        raise NotImplementedError("L32.5: implement LocalSerializedTensor.get")


def gather_handles_to_rank0(
    local_blob: bytes,
    rank: int,
    world_size: int,
    group: dict,
) -> Optional[List[bytes]]:
    """模拟 dist.gather_object：

    - rank != 0：把 local_blob 写进 group[rank]，返回 None
    - rank == 0：把 local_blob 写进 group[0]，等所有 rank 写完后返回
                 [group[0], group[1], ..., group[world_size-1]]

    本 CPU 模拟没有真分布式：每次调用先把 local_blob 写入 group[rank]。
      - rank != 0：返回 None
      - rank == 0：仅当 group 已收齐 world_size 个 entry 时返回完整列表；
                   否则返回 None（让调用方再调一次）。
    这样 rank 0 可以先注册自己的 blob，等其他 rank 全部就位后再次调用拿结果。
    """
    # TODO(student):
    #   group[rank] = local_blob
    #   if rank != 0:
    #       return None
    #   if len(group) < world_size:
    #       return None
    #   return [group[r] for r in range(world_size)]
    raise NotImplementedError("L32.5: implement gather_handles_to_rank0")


def update_weights_from_tensor(
    named_handles: List[Tuple[str, LocalSerializedTensor]],
    inference_state: Dict[str, torch.Tensor],
    tp_rank: int,
    pool: IPCStoragePool,
    flush_cache: bool = False,
) -> None:
    """模拟 SGLang ModelRunner.update_weights_from_tensor：

    每个 SGLang TP rank 调用本函数：
      - 对 named_handles 中每个 (name, lst)：
          - tensor = lst.get(tp_rank, pool)
          - inference_state[name] = tensor
      - 仅在 flush_cache=True 且 tensor_index == len(named_handles)-1 时清 pool
    """
    # TODO(student):
    #   n = len(named_handles)
    #   for tensor_index, (name, lst) in enumerate(named_handles):
    #       tensor = lst.get(tp_rank, pool)
    #       inference_state[name] = tensor
    #       if flush_cache and tensor_index == n - 1:
    #           pool.clear()
    raise NotImplementedError("L32.5: implement update_weights_from_tensor")
