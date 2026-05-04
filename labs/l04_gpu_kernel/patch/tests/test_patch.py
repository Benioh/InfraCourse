"""L01.7 Patch tests · GPU-only. Auto-skip on CPU."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.triton_softmax")


pytestmark = pytest.mark.gpu


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available — L01.7 needs GPU")


def test_matches_torch_softmax_fp32():
    _skip_if_no_cuda()
    impl = _impl()
    torch.manual_seed(42)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    expected = F.softmax(x, dim=-1)
    actual = impl.triton_softmax(x)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-5, f"max diff = {diff}"


def test_matches_torch_softmax_fp16():
    _skip_if_no_cuda()
    impl = _impl()
    torch.manual_seed(42)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
    expected = F.softmax(x, dim=-1)
    actual = impl.triton_softmax(x)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-2, f"max diff = {diff}"


def test_row_sums_to_one():
    _skip_if_no_cuda()
    impl = _impl()
    x = torch.randn(16, 512, device="cuda")
    y = impl.triton_softmax(x)
    sums = y.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_short_rows():
    _skip_if_no_cuda()
    impl = _impl()
    x = torch.randn(4, 64, device="cuda")
    expected = F.softmax(x, dim=-1)
    actual = impl.triton_softmax(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_long_rows():
    _skip_if_no_cuda()
    impl = _impl()
    x = torch.randn(2, 2048, device="cuda")
    expected = F.softmax(x, dim=-1)
    actual = impl.triton_softmax(x)
    assert torch.allclose(actual, expected, atol=1e-5)
