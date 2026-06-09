"""L14 Patch tests · CPU OK."""

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
    name = os.environ.get("IMPL") or "starter"
    return importlib.import_module(f"{name}.moe_router")


def test_each_token_routes_to_two():
    impl = _impl()
    torch.manual_seed(42)
    logits = torch.randn(64, 8)  # 64 tokens, 8 experts
    dispatch_mask, _, _ = impl.top2_router(logits, capacity_factor=4.0)
    # Without capacity overflow, every token has exactly 2 experts active
    counts = dispatch_mask.sum(dim=-1)
    assert torch.all(counts <= 2), f"max routes per token = {counts.max()}"
    assert torch.all(counts >= 2), f"with capacity_factor=4 no token should be dropped; min={counts.min()}"


def test_combine_weights_sum_to_one_no_capacity():
    impl = _impl()
    torch.manual_seed(42)
    logits = torch.randn(32, 4)
    _, weights, _ = impl.top2_router(logits, capacity_factor=10.0)
    sums = weights.sum(dim=-1)
    # All tokens routed → each row sums to ~1 (softmax over top-2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"weights sums (no capacity drop) should be ≈ 1; got min={sums.min()}, max={sums.max()}"
    )


def test_capacity_factor_drops_overflow():
    """capacity_factor=0.5 → each expert can hold only 0.5 * 2 / num_experts * num_tokens tokens.
    With 32 tokens / 4 experts: capacity = max(1, 0.5*32*2/4) = 8.
    Some experts are likely to overflow; total dispatched tokens must be < 32 * 2.
    """
    impl = _impl()
    torch.manual_seed(11)
    # Skew logits so one expert is highly favored — guarantees overflow
    logits = torch.randn(32, 4)
    logits[:, 0] += 5.0  # everyone wants expert 0

    dispatch_mask, _, _ = impl.top2_router(logits, capacity_factor=0.5)
    expert_loads = dispatch_mask.sum(dim=0)
    capacity = max(1, int(0.5 * 32 * 2 / 4))
    # No expert should exceed capacity
    assert torch.all(expert_loads <= capacity), (
        f"expert loads {expert_loads.tolist()} exceeded capacity {capacity}"
    )


def test_aux_loss_low_when_balanced():
    """For top-2 routing, the minimum (uniform) aux_loss is top_k=2.0,
    because Σ fraction_routed = top_k for top-k dispatch."""
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(256, 8) * 0.01  # near-uniform
    _, _, aux = impl.top2_router(logits, capacity_factor=10.0)
    assert aux.item() < 2.5, f"near-balanced top-2 aux_loss ≈ 2.0; got {aux.item():.3f}"


def test_aux_loss_high_when_imbalanced():
    """All tokens prefer expert 0 → aux_loss should be much greater than the
    balanced minimum (2.0 for top-2)."""
    impl = _impl()
    torch.manual_seed(0)
    num_experts = 8
    logits = torch.zeros(64, num_experts)
    logits[:, 0] = 100.0
    _, _, aux = impl.top2_router(logits, capacity_factor=10.0)
    assert aux.item() > 4.0, (
        f"imbalanced top-2 aux_loss should be much greater than 2.0; got {aux.item():.3f}"
    )
