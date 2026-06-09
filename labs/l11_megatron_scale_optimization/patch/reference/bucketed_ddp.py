"""Reference solution for L12 Patch · BucketedManualDDP."""

from __future__ import annotations

from typing import List, Optional

import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class BucketedManualDDP:
    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: float = 25,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.module = model
        self.process_group = process_group
        self.world_size = (
            dist.get_world_size(process_group) if dist.is_initialized() else 1
        )
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets = self._create_buckets()

    def _create_buckets(self) -> List[List[nn.Parameter]]:
        buckets: List[List[nn.Parameter]] = []
        current: List[nn.Parameter] = []
        current_size = 0
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            p_bytes = p.numel() * p.element_size()
            if current and current_size + p_bytes > self.bucket_size_bytes:
                buckets.append(current)
                current = []
                current_size = 0
            current.append(p)
            current_size += p_bytes
        if current:
            buckets.append(current)
        return buckets

    def synchronize_grads(self) -> None:
        if self.world_size == 1 or not dist.is_initialized():
            return
        for bucket in self.buckets:
            grads = [p.grad for p in bucket if p.grad is not None]
            if not grads:
                continue
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.process_group)
            flat /= self.world_size
            for g, ug in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(ug)
