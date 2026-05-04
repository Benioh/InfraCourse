"""L08.3 Patch · eager vs FlashAttention benchmark."""

from __future__ import annotations

import math
import time

import torch


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Reference O(N²) attention. q/k/v: [B, H, T, D]."""
    # TODO(student): scale = 1/sqrt(D)
    # TODO(student): scores = (q @ k.transpose(-2, -1)) * scale
    # TODO(student): if causal, build a [T, T] upper-triangular mask of -inf and add it
    # TODO(student): softmax dim=-1; matmul with v; return [B, H, T, D]
    raise NotImplementedError("L08.3: implement eager_attention")


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Use torch.nn.functional.scaled_dot_product_attention (FlashAttention backend on CUDA)."""
    # TODO(student): call F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    raise NotImplementedError("L08.3: implement flash_attention")


def bench_attention(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    device: str,
    dtype: torch.dtype = torch.float32,
    num_iters: int = 20,
) -> dict:
    """Benchmark eager vs flash and return timing/memory/correctness metrics."""
    # TODO(student): torch.manual_seed(0); allocate q/k/v on device
    # TODO(student): correctness pass in fp32 with seq_len capped (e.g. min(seq_len, 256)) on the device
    #               compute max_abs_diff between eager and flash outputs
    # TODO(student): perf pass in the requested dtype with full seq_len
    #   - warmup 5 iters
    #   - if device.startswith("cuda"): torch.cuda.synchronize() before/after; reset_peak; record peak
    #   - record total seconds for num_iters
    # TODO(student): return dict with eager_time_ms / flash_time_ms / speedup / max_abs_diff /
    #               peak_mem_eager_mb / peak_mem_flash_mb (CPU paths fill memory entries with 0.0)
    raise NotImplementedError("L08.3: implement bench_attention")
