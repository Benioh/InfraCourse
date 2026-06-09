"""
L12 Patch · Bucketed Manual DDP

填空规则：
- TODO(student) 必须自己写
- 不许用 torch.nn.parallel.DistributedDataParallel
- 允许 torch._utils._flatten_dense_tensors / _unflatten_dense_tensors

完成度自检：
    make patch-test M=l11_megatron_scale_optimization
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class BucketedManualDDP:
    """逐参数 ManualDDP 的进阶版：把 grad 按字节数分组合并 all-reduce."""

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
        # TODO(student): self.buckets = self._create_buckets()
        raise NotImplementedError("L12 Patch: implement __init__ buckets")

    def _create_buckets(self) -> List[List[nn.Parameter]]:
        """把 self.module 的 requires_grad params 按累加字节数分组。

        策略：按 named_parameters() 顺序，累加 (numel * element_size)；
        超过 bucket_size_bytes 就开新桶。注意：单 param > bucket_size 时
        也独占一桶（不要 split 一个 param）。
        """
        buckets: List[List[nn.Parameter]] = []
        current: List[nn.Parameter] = []
        current_size = 0
        # TODO(student):
        #   for p in self.module.parameters():
        #     if not p.requires_grad: skip
        #     p_bytes = p.numel() * p.element_size()
        #     if current and current_size + p_bytes > self.bucket_size_bytes:
        #         buckets.append(current); current=[]; current_size=0
        #     current.append(p); current_size += p_bytes
        #   if current: buckets.append(current)
        #   return buckets
        raise NotImplementedError("L12 Patch: implement _create_buckets")

    def synchronize_grads(self) -> None:
        if self.world_size == 1 or not dist.is_initialized():
            return
        # TODO(student):
        #   for bucket in self.buckets:
        #     grads = [p.grad for p in bucket if p.grad is not None]
        #     if not grads: continue
        #     flat = _flatten_dense_tensors(grads)
        #     dist.all_reduce(flat, op=SUM, group=self.process_group)
        #     flat /= self.world_size
        #     for g, ug in zip(grads, _unflatten_dense_tensors(flat, grads)):
        #         g.copy_(ug)
        raise NotImplementedError("L12 Patch: implement synchronize_grads")
