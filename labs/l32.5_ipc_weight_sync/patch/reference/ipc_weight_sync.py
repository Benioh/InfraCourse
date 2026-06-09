"""Reference solution for L37 Patch · CUDA IPC Weight Sync."""

from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


class IPCStoragePool:
    """CPU sim of CUDA IPC storage pool."""

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


def serialize_handle(tensor: torch.Tensor, pool: IPCStoragePool) -> bytes:
    handle = pool.put(tensor)
    meta = {
        "handle": handle,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
    }
    return pickle.dumps(meta)


def deserialize_handle(blob: bytes, pool: IPCStoragePool) -> torch.Tensor:
    meta = pickle.loads(blob)
    return pool.get(meta["handle"])


@dataclass
class LocalSerializedTensor:
    values: List[bytes]

    def get(self, rank: int, pool: IPCStoragePool) -> torch.Tensor:
        return deserialize_handle(self.values[rank], pool)


def gather_handles_to_rank0(
    local_blob: bytes,
    rank: int,
    world_size: int,
    group: dict,
) -> Optional[List[bytes]]:
    group[rank] = local_blob
    if rank != 0:
        return None
    # rank 0: return full list only when all ranks have registered; else None
    if len(group) < world_size:
        return None
    return [group[r] for r in range(world_size)]


def update_weights_from_tensor(
    named_handles: List[Tuple[str, LocalSerializedTensor]],
    inference_state: Dict[str, torch.Tensor],
    tp_rank: int,
    pool: IPCStoragePool,
    flush_cache: bool = False,
) -> None:
    n = len(named_handles)
    for tensor_index, (name, lst) in enumerate(named_handles):
        tensor = lst.get(tp_rank, pool)
        inference_state[name] = tensor
        if flush_cache and tensor_index == n - 1:
            pool.clear()
