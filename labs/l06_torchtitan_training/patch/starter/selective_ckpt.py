"""
L07 Patch · Selective Activation Checkpoint

填空规则：
- TODO(student) 必须自己写
- 不许用 fairscale / deepspeed
- 允许 torch.utils.checkpoint 全部 API

完成度自检：
    make patch-test M=l06_torchtitan_training
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


PolicyFn = Callable[[str, nn.Module], bool]


class _CheckpointWrapper(nn.Module):
    """把一个 child module 包起来，forward 用 torch.utils.checkpoint."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # use_reentrant=False 是 PyTorch >= 2.1 推荐写法；老版本可改 True
        return checkpoint(self.module, *args, use_reentrant=False, **kwargs)


def selective_checkpoint_wrap(model: nn.Module, policy_fn: PolicyFn) -> nn.Module:
    """对 model.named_children() 中 policy_fn(name, child) 返回 True 的 child，
    用 _CheckpointWrapper 包起来。in-place 修改 model 后返回。

    注意：这里只对一层 child 操作；嵌套模型需要递归调用。
    """
    # TODO(student):
    #   遍历 model.named_children()，对每个 (name, child):
    #     if policy_fn(name, child):
    #       wrapped = _CheckpointWrapper(child)
    #       setattr(model, name, wrapped)
    #   return model
    raise NotImplementedError("L07 Patch: implement selective_checkpoint_wrap")


def attention_only_policy(name: str, module: nn.Module) -> bool:
    """只对名字含 'attn' / 'attention' 的子模块返回 True."""
    # TODO(student):
    #   return 'attn' in name.lower() or 'attention' in name.lower()
    raise NotImplementedError("L07 Patch: implement attention_only_policy")
