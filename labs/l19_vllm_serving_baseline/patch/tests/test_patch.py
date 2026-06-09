"""L20 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.typical_p")


def test_typical_p_one_keeps_all():
    impl = _impl()
    torch.manual_seed(42)
    logits = torch.randn(2, 100)
    out = impl.typical_p_filter(logits, typical_p=1.0)
    assert torch.equal(out, logits)


def test_typical_p_zero_keeps_one():
    """typical_p ≈ 0 → only 1 token kept (the one closest to entropy)."""
    impl = _impl()
    torch.manual_seed(42)
    logits = torch.randn(1, 50)
    out = impl.typical_p_filter(logits, typical_p=1e-6)
    n_kept = int((out > -float("inf")).sum().item())
    assert n_kept == 1, f"expected exactly 1 kept; got {n_kept}"


def test_filter_value_applied():
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(1, 30)
    out = impl.typical_p_filter(logits, typical_p=0.5, filter_value=-1e9)
    # Removed positions equal exactly filter_value
    removed = out == -1e9
    assert removed.any(), "some tokens should have been removed"
    # Kept positions retain original logit values
    kept_mask = ~removed
    assert torch.allclose(out[kept_mask], logits[kept_mask])


def test_kept_tokens_dist_minimal():
    """Tokens kept should have smaller |info - entropy| than dropped ones."""
    impl = _impl()
    torch.manual_seed(7)
    logits = torch.randn(1, 100)
    probs = torch.softmax(logits, dim=-1)
    info = -torch.log(probs + 1e-10)
    entropy = (probs * info).sum(dim=-1, keepdim=True)
    dist = (info - entropy).abs()  # (1, 100)

    out = impl.typical_p_filter(logits, typical_p=0.5)
    kept_mask = out > -float("inf")  # (1, 100)
    if kept_mask.sum() > 0 and (~kept_mask).sum() > 0:
        max_kept_dist = dist[kept_mask].max().item()
        min_dropped_dist = dist[~kept_mask].min().item()
        assert max_kept_dist <= min_dropped_dist + 1e-6, (
            f"kept set should have smaller distance; "
            f"max kept = {max_kept_dist}, min dropped = {min_dropped_dist}"
        )


def test_works_with_batch():
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    out = impl.typical_p_filter(logits, typical_p=0.6)
    assert out.shape == logits.shape
    # Each batch row independently has at least 1 token kept
    for b in range(4):
        n = int((out[b] > -float("inf")).sum().item())
        assert n >= 1, f"batch {b} has 0 tokens kept"
