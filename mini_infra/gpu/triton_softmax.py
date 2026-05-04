"""
Online softmax 数学骨架（L01.7 教学）。

教学目的：
    Triton fused softmax 内部就是 online softmax：把一行分块加载到 SMEM，
    每块更新 (running_max, running_sum)，最后一次性输出。这里用纯 Python
    复现这个算法骨架，让学员能在 CPU 上跑通、断言、求 max_abs_err，再去
    写真正的 .triton 实现时不会被数学搞糊涂。

真实框架对照：
    - github_repo/triton/python/tutorials/02-fused-softmax.py
        kernel 内 `tl.load(... mask=...)` 对应 mask_needed；
        `tl.exp(...)` + `tl.sum(...)` 对应 online_softmax 的累加。
    - github_repo/Megatron-LM/megatron/core/fusions/fused_softmax.py
        生产 backend 在数值稳定性之外还做 dtype dispatch 与 mask broadcast，
        但核心数学不变。

简化掉的复杂度：
    - 没有 GPU 并行，单线程逐元素
    - 没有 dtype 选择，全 float64
    - 不处理 attention mask（causal/padding mask）
"""
from __future__ import annotations

import math
from typing import Iterable

from .memory_model import roofline_softmax


def stable_softmax(row: Iterable[float]) -> list[float]:
    """一次性数值稳定 softmax，用作 online_softmax 的参考实现。"""

    values = list(row)
    if not values:
        return []
    row_max = max(values)
    exp_values = [math.exp(value - row_max) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def online_softmax(row: Iterable[float], block_size: int) -> list[float]:
    """块累加版 softmax，等价于 Triton kernel 的算法。

    关键不变量：
        running_sum 必须用 `exp(running_max - new_max)` 重缩放，否则
        切换到更大的 new_max 后旧的指数项就被低估。删掉这一步会在
        长序列 + fp16 下数值崩塌（max_abs_err > 1e-2）。
    """

    values = list(row)
    running_max = -math.inf
    running_sum = 0.0
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        block_max = max(block) if block else -math.inf
        new_max = max(running_max, block_max)
        running_sum = running_sum * math.exp(running_max - new_max) + sum(
            math.exp(value - new_max) for value in block
        )
        running_max = new_max
    return [math.exp(value - running_max) / running_sum for value in values]


def max_abs_error(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    pairs = zip(lhs, rhs, strict=False)
    return max((abs(left - right) for left, right in pairs), default=0.0)


def mask_needed(seq_len: int, block_size: int) -> bool:
    """tail block 是否会读越界。

    在 Triton 中对应 `tl.load(ptr, mask=offsets < seq_len, other=-inf)`。
    如果 seq_len 不是 BLOCK_SIZE 的整数倍，最后一个 block 必须用 mask
    把超出 seq_len 的位置填 -inf，否则 online_max 会被垃圾值污染。
    """



def simulate_softmax_kernel(
    batch: int = 8,
    seq_len: int = 4096,
    block_size: int = 1024,
    device: str = "rtx4090",
) -> dict[str, float | int | str | bool]:
    sample = [((index * 17) % 97 - 48) / 8 for index in range(min(seq_len, 2048))]
    reference = stable_softmax(sample)
    candidate = online_softmax(sample, block_size=max(1, min(block_size, len(sample))))
    estimate = roofline_softmax(batch, seq_len, block_size, device=device)
    estimate.update(
        {
            "kernel": "online_fused_softmax",
            "mask_needed": mask_needed(seq_len, block_size),
            "max_abs_err": round(max_abs_error(reference, candidate), 12),
            "tl_load_mask_reason": "tail block may read past seq_len when seq_len is not divisible by BLOCK_SIZE",
        }
    )
    return estimate
