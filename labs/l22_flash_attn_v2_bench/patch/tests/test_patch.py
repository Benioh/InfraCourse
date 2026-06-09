"""L23 Patch tests."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.flash_bench")


def test_eager_shape_correct():
    impl = _impl()
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    out = impl.eager_attention(q, k, v, causal=False)
    assert out.shape == q.shape


def test_eager_causal_mask_no_leak():
    impl = _impl()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 1, 4, 4).contiguous().float()
    v = torch.cat([v, torch.zeros(1, 1, 4, 4)], dim=-1)
    out = impl.eager_attention(q, k, v, causal=True)
    # Row 0 may only attend to position 0 → first 4 cols are e0
    row0 = out[0, 0, 0]
    assert torch.allclose(row0[:4], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_flash_matches_eager_numerically():
    impl = _impl()
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 8, dtype=torch.float32)
    out_e = impl.eager_attention(q, k, v, causal=True)
    out_f = impl.flash_attention(q, k, v, causal=True)
    assert torch.allclose(out_e, out_f, atol=1e-5, rtol=1e-5)


def test_bench_returns_required_metrics():
    impl = _impl()
    out = impl.bench_attention(
        seq_len=64, num_heads=2, head_dim=8, causal=True, device="cpu", dtype=torch.float32, num_iters=2
    )
    for key in [
        "eager_time_ms",
        "flash_time_ms",
        "speedup",
        "max_abs_diff",
        "peak_mem_eager_mb",
        "peak_mem_flash_mb",
    ]:
        assert key in out
    assert out["max_abs_diff"] < 1e-4


@pytest.mark.gpu
def test_bench_speedup_gpu():
    impl = _impl()
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    out = impl.bench_attention(
        seq_len=2048, num_heads=8, head_dim=64, causal=True, device="cuda",
        dtype=torch.bfloat16, num_iters=10,
    )
    assert out["speedup"] >= 1.5


@pytest.mark.gpu
def test_bench_peak_memory_gpu():
    impl = _impl()
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    out = impl.bench_attention(
        seq_len=2048, num_heads=8, head_dim=64, causal=True, device="cuda",
        dtype=torch.bfloat16, num_iters=5,
    )
    if out["peak_mem_eager_mb"] > 0:
        assert out["peak_mem_flash_mb"] <= out["peak_mem_eager_mb"]
