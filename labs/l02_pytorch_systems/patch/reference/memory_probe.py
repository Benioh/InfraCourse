"""Reference solution for L01 Patch · 显存账本."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def count_grad_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.numel() * p.grad.element_size()
    return total


def count_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total
