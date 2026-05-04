"""Reference solution for L08.3 Patch."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F  # noqa: N812


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        T = scores.size(-1)
        mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _alloc(B, H, T, D, device, dtype):
    return torch.randn(B, H, T, D, device=device, dtype=dtype)


def _peak_mem_mb(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _time_iters(fn, q, k, v, causal, num_iters, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn(q, k, v, causal)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / num_iters


def bench_attention(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    device: str,
    dtype: torch.dtype = torch.float32,
    num_iters: int = 20,
) -> dict:
    torch.manual_seed(0)
    B = 1
    # correctness on a smaller tensor in fp32
    cap = min(seq_len, 256)
    q32 = torch.randn(B, num_heads, cap, head_dim, device=device, dtype=torch.float32)
    k32 = torch.randn(B, num_heads, cap, head_dim, device=device, dtype=torch.float32)
    v32 = torch.randn(B, num_heads, cap, head_dim, device=device, dtype=torch.float32)
    out_eager = eager_attention(q32, k32, v32, causal)
    out_flash = flash_attention(q32, k32, v32, causal)
    max_abs_diff = float((out_eager - out_flash).abs().max())

    q = _alloc(B, num_heads, seq_len, head_dim, device, dtype)
    k = _alloc(B, num_heads, seq_len, head_dim, device, dtype)
    v = _alloc(B, num_heads, seq_len, head_dim, device, dtype)

    for _ in range(5):
        eager_attention(q, k, v, causal)
        flash_attention(q, k, v, causal)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    eager_ms = _time_iters(eager_attention, q, k, v, causal, num_iters, device)
    eager_peak = _peak_mem_mb(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    flash_ms = _time_iters(flash_attention, q, k, v, causal, num_iters, device)
    flash_peak = _peak_mem_mb(device)

    return {
        "eager_time_ms": eager_ms,
        "flash_time_ms": flash_ms,
        "speedup": eager_ms / max(flash_ms, 1e-9),
        "max_abs_diff": max_abs_diff,
        "peak_mem_eager_mb": eager_peak,
        "peak_mem_flash_mb": flash_peak,
    }
