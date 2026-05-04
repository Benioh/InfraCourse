"""Reference solution for L03 Patch · Selective Activation Checkpoint."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

PolicyFn = Callable[[str, nn.Module], bool]


class _CheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, use_reentrant=False, **kwargs)


def selective_checkpoint_wrap(model: nn.Module, policy_fn: PolicyFn) -> nn.Module:
    for name, child in list(model.named_children()):
        if policy_fn(name, child):
            setattr(model, name, _CheckpointWrapper(child))
    return model


def attention_only_policy(name: str, module: nn.Module) -> bool:
    n = name.lower()
    return "attn" in n or "attention" in n
