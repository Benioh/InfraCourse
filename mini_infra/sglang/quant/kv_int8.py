"""
SGLang KV-cache INT8 量化骨架（L08.5 教学）。

教学目的：
    把 KV cache 的对称量化具体化（symmetric int8: scale = max|x|/127）。
    学员能看到：
        - fp16 KV 与 int8 KV 的字节差（约 2x）
        - dequant 的 max_abs_err 量级
        - radix prefix cache 兼容性边界：cache key 还是 token-based，
          但 stored block 需要 scale metadata 才能 dequant

真实框架对照：
    - github_repo/sglang/python/sglang/srt/layers/quantization/
        SGLang 的 KV-cache 量化实现（含 fp8 与 int8 选项）
    - github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py
        radix cache 对量化 KV 的兼容（key 不变，value 存量化值 + scale）

简化掉的复杂度：
    - 不做 per-head / per-token / per-channel 区分（只做 per-tensor）
    - 不做异步 dequant（真实有的实现把 dequant 放到 GEMM 内）
    - 不验证 prefix cache 命中后的数值漂移
"""
from __future__ import annotations


def symmetric_scale(values: list[float]) -> float:
    """对称量化 scale：max(|x|) / 127。

    用 127 而非 128 是为了让最负值也能精确表示（int8 范围 [-128, 127]）。
    非对称量化用 (max - min)/255 + zero point，对 KV cache 通常不必要
    （KV 分布大致零中心）。
    """
    return round(max(max(abs(value) for value in values), 1e-9) / 127, 8)


def quantize_kv(values: list[float]) -> tuple[list[int], float]:
    scale = symmetric_scale(values)
    return [max(-127, min(127, round(value / scale))) for value in values], scale


def dequantize_kv(values: list[int], scale: float) -> list[float]:
    return [round(value * scale, 6) for value in values]


def kv_int8_summary(tokens: int = 4096, heads: int = 32, head_dim: int = 128) -> dict[str, object]:
    sample = [-1.25, -0.2, 0.0, 0.35, 1.1]
    quantized, scale = quantize_kv(sample)
    fp16_bytes = tokens * heads * head_dim * 2 * 2
    int8_bytes = tokens * heads * head_dim * 2 + heads * 4
    restored = dequantize_kv(quantized, scale)
    return {
        "quant": "kv_int8_symmetric",
        "tokens": tokens,
        "heads": heads,
        "head_dim": head_dim,
        "scale": scale,
        "quantized": quantized,
        "max_abs_err": round(max(abs(a - b) for a, b in zip(sample, restored, strict=False)), 6),
        "fp16_kv_mem_gb": round(fp16_bytes / 1e9, 6),
        "int8_kv_mem_gb": round(int8_bytes / 1e9, 6),
        "cache_compatible_boundary": "radix cache keys remain token/prefix based; stored KV blocks need scale metadata",
    }
