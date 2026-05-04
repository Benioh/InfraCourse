"""Reference solution for L01.5 Patch · ManualDDP."""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist
import torch.nn as nn


class ManualDDP:
    def __init__(self, model: nn.Module, process_group: Optional[dist.ProcessGroup] = None):
        self.module = model
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1

    def synchronize_grads(self) -> None:
        if self.world_size == 1 or not dist.is_initialized():
            return
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.process_group)
            p.grad /= self.world_size
