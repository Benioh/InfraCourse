"""
量化 calibration（SmoothQuant 一阶 + AWQ activation order）（L08.5 教学）。

教学目的：
    把 "为什么需要 calibration set" 具体化：fp8/AWQ 都需要从一组真实激活
    样本算出 scale/order，否则量化后 acc 大跌。学员在 n17 notebook 里：
        - 看 channel_absmax 是 SmoothQuant 的输入信号
        - 看 alpha=0.5 把激活异常值"挪一半到权重"
        - 看 awq_activation_order 按激活幅度排序，决定哪些 channel 先被
          保护（更高 group precision）

真实框架对照：
    - github_repo/llm-compressor (前 SparseML)：SmoothQuant 与 AWQ 的
      生产级实现；本文件只保留一阶简化。
    - SmoothQuant 论文（Xiao et al., 2022）；AWQ 论文（Lin et al., 2023）。

简化掉的复杂度：
    - alpha 实际可学（按 channel 或 layer），本文件固定 0.5
    - 真实 calibration set 通常 128-512 个 sample，本文件用 3 个示意
    - 不做 activation reordering 后的权重重排
"""
from __future__ import annotations


def channel_absmax(samples: list[list[float]]) -> list[float]:
    """对 calibration samples 按 channel（最后一维）求 |x| 的最大值。

    这是 SmoothQuant 与 AWQ 共同的输入信号：channel 上 |x| 越大说明
    该 channel 激活异常值越多，量化时需要特殊处理。
    """
    if not samples:
        return []
    width = len(samples[0])
    return [max(abs(row[index]) for row in samples) for index in range(width)]


def smoothquant_scales(samples: list[list[float]], alpha: float = 0.5) -> list[float]:
    maxima = channel_absmax(samples)
    return [round(max(value, 1e-6) ** alpha, 6) for value in maxima]


def awq_activation_order(samples: list[list[float]]) -> list[int]:
    maxima = channel_absmax(samples)
    return [index for index, _ in sorted(enumerate(maxima), key=lambda item: item[1], reverse=True)]


def calibration_summary() -> dict[str, object]:
    samples = [
        [0.2, -1.0, 3.0, 0.1],
        [0.5, -0.2, 2.2, 0.4],
        [0.1, -1.5, 4.0, -0.3],
    ]
    scales = smoothquant_scales(samples, alpha=0.5)
    return {
        "samples": len(samples),
        "smoothquant_alpha": 0.5,
        "smoothquant_scales": scales,
        "awq_activation_order": awq_activation_order(samples),
        "calibration_note": "Scales are solved from activation statistics; real kernels must validate accuracy drift.",
    }
