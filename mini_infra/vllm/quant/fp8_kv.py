"""
FP8 (e4m3) 量化骨架（L08.5 教学）。

教学目的：
    把 fp8 的"求 scale → quantize → dequantize"三步具体化，让学员在
    nontrivial sample 上看到 max_abs_err，理解 fp8 不是"无损压缩"。

教学要点：
    - e4m3 范围 ±448（fp8_max=448）
    - per-tensor scale = max_abs / fp8_max
    - quantize 后值范围限到 [-448, +448]
    - dequant 后误差与原始值幅度成正比

真实框架对照：
    - github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py
        真实 fp8 量化包含 calibration、per-channel/per-tensor 切换、
        e4m3/e5m2 选择、CUDA kernel；本文件只做数学骨架。
    - H100/H200 才有原生 fp8 支持；4090 上跑只能软件模拟（极慢）。

简化掉的复杂度：
    - 不做 e4m3 vs e5m2 的范围选择
    - 不做 per-channel scale
    - 不调 CUDA fp8 kernel；纯 Python int 模拟
"""
from __future__ import annotations


def fp8_scale(max_abs: float, fp8_max: float = 448.0) -> float:
    """求 per-tensor fp8 scale。

    fp8_max=448 对应 e4m3。e5m2 的 max 是 57344，范围更大但精度更低。
    实际计算时用 max(|x|) 而不是 std/3σ，因为 fp8 范围本来就很窄。
    """
    return round(max(max_abs, 1e-9) / fp8_max, 8)


def quantize_to_fp8(values: list[float], scale: float) -> list[int]:
    return [max(-448, min(448, round(value / scale))) for value in values]


def dequantize_from_fp8(values: list[int], scale: float) -> list[float]:
    return [round(value * scale, 6) for value in values]


def fp8_summary(values: list[float] | None = None) -> dict[str, object]:
    data = values or [-2.0, -0.5, 0.0, 0.75, 3.0]
    scale = fp8_scale(max(abs(value) for value in data))
    quantized = quantize_to_fp8(data, scale)
    restored = dequantize_from_fp8(quantized, scale)
    max_error = max(abs(left - right) for left, right in zip(data, restored, strict=False))
    return {
        "quant": "fp8_e4m3_sim",
        "scale": scale,
        "quantized": quantized,
        "restored": restored,
        "max_abs_err": round(max_error, 6),
        "h200_only_for_real_kernel": True,
    }
