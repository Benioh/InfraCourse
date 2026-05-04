"""
L01 Patch · 显存账本

填空规则：
- 凡是 `# TODO(student): ...` 必须自己写。
- 不许用 torchsummary / torchinfo 等第三方库。
- 允许 torch.* 所有内置 API。

完成度自检：
    make patch-test M=l02_pytorch_systems
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_param_bytes(model: nn.Module) -> int:
    """所有 nn.Parameter 占用的字节数。

    提示：每个 tensor 的字节数 = `t.numel() * t.element_size()`。
    fp32: element_size=4，fp16/bf16: 2，int8: 1。
    """
    # TODO(student): 遍历 model.parameters()，把 p.numel() * p.element_size() 全加起来。
    raise NotImplementedError("L01 Patch: implement count_param_bytes")


def count_grad_bytes(model: nn.Module) -> int:
    """所有 `.grad` 占用的字节数。

    陷阱：
    - backward 之后每个 param 都有 grad；
    - 但 `optimizer.zero_grad(set_to_none=True)` 之后 grad=None，要跳过；
    - p.grad 的 dtype 通常与 p 一致（混合精度训练例外）。
    """
    # TODO(student):
    #   遍历 model.parameters()，对每个 p:
    #     if p.grad is not None: 累加 p.grad.numel() * p.grad.element_size()
    raise NotImplementedError("L01 Patch: implement count_grad_bytes")


def count_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """所有 `optimizer.state[p]` 中 Tensor 类型 entry 的字节数。

    optimizer.state 是 `{param: {state_name: state_value}}` 的 dict。
    state_value 可能是 Tensor（如 Adam 的 exp_avg），也可能是 Python int（step counter），
    我们只统计 Tensor 部分。

    经验值：
      - plain SGD：0
      - SGD with momentum：≈ 1 × param_bytes（momentum_buffer）
      - Adam：≈ 2 × param_bytes（exp_avg + exp_avg_sq）
      - AdamW：同 Adam
    """
    # TODO(student):
    #   遍历 optimizer.state.values()（每个 v 是一个 dict）；
    #   再遍历 v.values()，只对 isinstance(x, torch.Tensor) 累加 numel * element_size。
    raise NotImplementedError("L01 Patch: implement count_optimizer_state_bytes")
