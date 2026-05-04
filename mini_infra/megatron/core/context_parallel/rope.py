"""
RoPE（Rotary Position Embedding）最小同构（L04.5 教学）。

教学目的：
    用纯 Python 复现 RoPE 的频率公式与 Q/K 旋转，让学员在 n13 notebook
    里能：
        - 验证 inv_freq[i] = base^(-2i/d) 的几何衰减
        - 验证 Q·K 内积只依赖位置差 (m-n)，不依赖绝对位置
        - 看到 base=10000 在长 seq 上的高频位"绕圈"现象（外推失效根源）
    再去 yarn.py 看 YaRN 如何缩放频率以缓解外推。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/models/common/embeddings/
      rotary_pos_embedding.py：真实 RoPE，含 fused kernel、cache、不同
      dtype/layout 兼容。本文件只保留数学骨架。

简化掉的复杂度：
    - 没有 GPU kernel；纯 Python 逐元素
    - 没有 fp16/bf16；全用 float64
    - 没有 cache 复用（真实实现会 cache cos/sin）
"""
from __future__ import annotations

import math


def inverse_frequencies(dim: int, base: float = 10000.0) -> list[float]:
    """RoPE 反频率：inv_freq[i] = base^(-2i/d)，i ∈ [0, d/2)。

    几何含义：低维位（i 小）频率高、相位变化快；高维位（i 大）频率低、
    相位变化慢。base=10000 时 d=128 的最高维位在 4K seq 上仍有空间，
    但推到 32K 时就会"绕圈"——这正是 YaRN/PI 要解决的问题。
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    return [base ** (-index / dim) for index in range(0, dim, 2)]


def rope_angles(seq_len: int, dim: int, base: float = 10000.0) -> list[list[float]]:
    freqs = inverse_frequencies(dim, base)
    return [[position * freq for freq in freqs] for position in range(seq_len)]


def rotate_pair(x_even: float, x_odd: float, angle: float) -> tuple[float, float]:
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return x_even * cos_value - x_odd * sin_value, x_even * sin_value + x_odd * cos_value


def apply_rope_pairs(values: list[float], position: int, base: float = 10000.0) -> list[float]:
    if len(values) % 2 != 0:
        raise ValueError("RoPE input length must be even")
    freqs = inverse_frequencies(len(values), base)
    rotated: list[float] = []
    for pair_index, freq in enumerate(freqs):
        even, odd = values[2 * pair_index], values[2 * pair_index + 1]
        rotated.extend(rotate_pair(even, odd, position * freq))
    return rotated


def rope_summary(
    seq_len: int = 4096, dim: int = 128, base: float = 10000.0
) -> dict[str, float | int]:
    freqs = inverse_frequencies(dim, base)
    return {
        "seq_len": seq_len,
        "dim": dim,
        "base": base,
        "rope_freq_min": round(min(freqs), 10),
        "rope_freq_max": round(max(freqs), 10),
        "max_angle": round((seq_len - 1) * max(freqs), 6),
    }
