"""L24 Patch tests · CPU OK."""

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
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.awq_calibrate")


def test_int8_range():
    impl = _impl()
    torch.manual_seed(42)
    W = torch.randn(64, 128)
    act_amax = torch.randn(128).abs() + 0.1
    scale = impl.compute_awq_scale(W, act_amax, alpha=0.5)
    int_w, per_out_scale = impl.quantize_w8_per_channel(W, scale)
    assert int_w.dtype == torch.int8
    assert (int_w >= -127).all()
    assert (int_w <= 127).all()


def test_quantize_dequantize_roundtrip():
    impl = _impl()
    torch.manual_seed(42)
    W = torch.randn(32, 64) * 0.5
    act_amax = torch.ones(64)
    scale = impl.compute_awq_scale(W, act_amax, alpha=0.5)
    int_w, per_out_scale = impl.quantize_w8_per_channel(W, scale)
    W_hat = impl.dequantize_w8_per_channel(int_w, per_out_scale, awq_scale=scale)
    rel = (W - W_hat).abs().mean() / (W.abs().mean() + 1e-9)
    assert rel.item() < 0.05, f"relative error = {rel.item():.4f}; expected < 0.05"


def test_per_channel_scale_shape():
    impl = _impl()
    W = torch.randn(64, 128)
    act_amax = torch.randn(128).abs() + 0.1
    scale = impl.compute_awq_scale(W, act_amax)
    assert scale.shape == (128,), f"expected (128,); got {scale.shape}"


def test_awq_scale_responds_to_outliers():
    """AWQ scale should be > 1 on outlier-activation channels (algorithm is doing something).

    Note: AWQ's quality benefit appears on real LLM weights (salient channel structure),
    not synthetic Gaussian. Here we just verify the scaling logic responds to outliers.
    """
    impl = _impl()
    torch.manual_seed(42)
    W = torch.randn(64, 128) * 0.1
    # Inject activation outliers in 5 channels (10x bigger)
    act_amax = torch.ones(128)
    act_amax[::20] *= 10  # outlier channels at indices 0, 20, 40, ...

    scale = impl.compute_awq_scale(W, act_amax, alpha=0.5)

    # Outlier channel scales should be larger than non-outlier
    outlier_scales = scale[::20]
    nonoutlier_scales = scale[10::20]  # offset by 10 to avoid outlier indices
    assert outlier_scales.mean() > nonoutlier_scales.mean(), (
        f"AWQ scale should be larger on outlier channels; "
        f"outlier_mean={outlier_scales.mean():.3f}, non_outlier_mean={nonoutlier_scales.mean():.3f}"
    )

    # And scale should be ≥ 1 everywhere (we clamp from below)
    assert (scale >= 1.0 - 1e-6).all()

    # End-to-end roundtrip with AWQ scale should still have reasonable error
    int_w, per_out = impl.quantize_w8_per_channel(W, scale)
    W_hat = impl.dequantize_w8_per_channel(int_w, per_out, awq_scale=scale)
    rel = (W - W_hat).abs().mean() / (W.abs().mean() + 1e-9)
    assert rel.item() < 0.1, f"AWQ roundtrip rel err = {rel.item():.4f}; expected < 0.1"


def test_alpha_zero_equals_naive():
    """alpha=0 → scale = ones → equivalent to plain per-output-channel quant."""
    impl = _impl()
    torch.manual_seed(7)
    W = torch.randn(16, 32)
    act_amax = torch.randn(32).abs() + 0.5
    scale = impl.compute_awq_scale(W, act_amax, alpha=0.0)
    # With alpha=0, scale should be all 1.0 (clamped from below)
    assert torch.allclose(scale, torch.ones_like(scale), atol=1e-5)
