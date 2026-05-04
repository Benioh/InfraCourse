"""L04.5 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import math
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
    return importlib.import_module(f"{name}.ring_attention")


def _full_attention(q, k, v):
    """Reference: F.scaled_dot_product_attention without causal mask."""
    return F.scaled_dot_product_attention(q, k, v)


def test_matches_full_attention_one_chunk():
    impl = _impl()
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    expected = _full_attention(q, k, v)
    actual = impl.ring_attention_forward(q, k, v, num_chunks=1)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-4, f"max diff = {diff}"


def test_matches_full_attention_four_chunks():
    impl = _impl()
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    expected = _full_attention(q, k, v)
    actual = impl.ring_attention_forward(q, k, v, num_chunks=4)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-4, f"max diff = {diff}"


def test_handles_uneven_chunks():
    """Sk=7 split into 2 chunks → [4, 3] (torch.chunk default)."""
    impl = _impl()
    torch.manual_seed(7)
    q = torch.randn(1, 2, 5, 16)
    k = torch.randn(1, 2, 7, 16)
    v = torch.randn(1, 2, 7, 16)
    expected = _full_attention(q, k, v)
    actual = impl.ring_attention_forward(q, k, v, num_chunks=2)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-4, f"max diff = {diff}"


def test_long_seq():
    impl = _impl()
    torch.manual_seed(11)
    q = torch.randn(1, 2, 256, 32)
    k = torch.randn(1, 2, 1024, 32)
    v = torch.randn(1, 2, 1024, 32)
    expected = _full_attention(q, k, v)
    actual = impl.ring_attention_forward(q, k, v, num_chunks=8)
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-3, f"max diff = {diff}"


def test_grads_are_continuous():
    """backward should run without error and produce finite grads."""
    impl = _impl()
    torch.manual_seed(13)
    q = torch.randn(1, 2, 8, 16, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, requires_grad=True)
    v = torch.randn(1, 2, 8, 16, requires_grad=True)
    out = impl.ring_attention_forward(q, k, v, num_chunks=2)
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all(), f"grad has nan/inf"
