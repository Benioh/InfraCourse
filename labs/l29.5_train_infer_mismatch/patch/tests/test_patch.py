"""L32 Patch tests · CPU OK.

Switch starter / reference via env var:  IMPL=reference make patch-test
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.mismatch")


def test_k3_kl_zero_when_aligned():
    impl = _impl()
    logp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    k3 = impl.compute_k3_kl(logp, logp)
    assert k3.abs().item() < 1e-6, f"K3 should be 0 when aligned, got {k3.item()}"


def test_k3_kl_formula():
    """logp = -1, logq = -2  →  log_ratio = 1, ratio = e
    K3 = e - 1 - 1 = e - 2 ≈ 0.7183
    """
    impl = _impl()
    logp = torch.tensor([-1.0])
    logq = torch.tensor([-2.0])
    k3 = impl.compute_k3_kl(logp, logq).item()
    expected = math.e - 1.0 - 1.0
    assert abs(k3 - expected) < 1e-5, f"K3 formula off: got {k3}, expected {expected}"


def test_k3_kl_nonnegative():
    impl = _impl()
    torch.manual_seed(0)
    for _ in range(20):
        logp = torch.randn(64) * 2
        logq = torch.randn(64) * 2
        k3 = impl.compute_k3_kl(logp, logq).item()
        assert k3 >= -1e-6, f"K3 should be non-negative, got {k3}"


def test_tis_clips_high_ratio():
    """ratio = exp(0 - (-2)) = e^2 ≈ 7.39, should be clipped to hi=2.0."""
    impl = _impl()
    logp_old = torch.tensor([-2.0])
    logp_new = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    out = impl.tis_correct(logp_old, logp_new, adv, lo=0.5, hi=2.0)
    assert abs(out.item() - 2.0) < 1e-5, f"TIS clip failed: {out.item()}"


def test_tis_passes_through_in_range():
    impl = _impl()
    logp_old = torch.tensor([-1.0])
    logp_new = torch.tensor([-1.0])  # ratio = 1.0
    adv = torch.tensor([3.0])
    out = impl.tis_correct(logp_old, logp_new, adv, lo=0.5, hi=2.0).item()
    assert abs(out - 3.0) < 1e-5


def test_mis_masks_outliers():
    """Token 0: ratio = e^5 ≈ 148, MIS should zero it.
    Token 1: ratio = 1, MIS should keep it.
    """
    impl = _impl()
    logp_old = torch.tensor([-5.0, -1.0])
    logp_new = torch.tensor([0.0, -1.0])
    adv = torch.tensor([1.0, 1.0])
    out = impl.mis_with_mask(logp_old, logp_new, adv, lo=0.5, hi=2.0)
    assert out[0].abs().item() < 1e-6, f"MIS should mask outlier, got {out[0]}"
    assert abs(out[1].item() - 1.0) < 1e-5, f"MIS in-range token wrong: {out[1]}"


def test_geometric_seq_is_matches_definition():
    """log_ratios = [[0.1, 0.2, 0.3, 0], [0.5, -0.5, 0, 0]]
    seq 0 (len=3): mean = (0.1+0.2+0.3)/3 = 0.2 → exp(0.2)
    seq 1 (len=2): mean = (0.5-0.5)/2 = 0 → exp(0) = 1.0
    """
    impl = _impl()
    logp_old = torch.zeros(2, 4)
    logp_new = torch.tensor([[0.1, 0.2, 0.3, 99.0], [0.5, -0.5, 99.0, 99.0]])
    seq_lens = torch.tensor([3, 2])
    out = impl.geometric_seq_is(logp_old, logp_new, seq_lens)
    assert out.shape == (2,)
    assert abs(out[0].item() - math.exp(0.2)) < 1e-5
    assert abs(out[1].item() - 1.0) < 1e-5


def test_veto_drops_extreme_low_prob():
    impl = _impl()
    logp = torch.tensor([math.log(1e-3), math.log(1e-7), math.log(0.5)])
    mask = impl.apply_veto(logp, threshold=1e-6)
    assert mask[0].item() == 1.0  # 1e-3 > 1e-6 → keep
    assert mask[1].item() == 0.0  # 1e-7 < 1e-6 → veto
    assert mask[2].item() == 1.0  # 0.5 > 1e-6 → keep


def test_batch_normalize_mean_one():
    impl = _impl()
    torch.manual_seed(0)
    weights = torch.rand(128) * 5 + 0.1
    out = impl.batch_normalize_weights(weights)
    assert abs(out.mean().item() - 1.0) < 1e-5, f"Mean after norm: {out.mean().item()}"


def test_batch_normalize_preserves_relative_order():
    impl = _impl()
    weights = torch.tensor([0.5, 1.0, 2.0, 4.0])
    out = impl.batch_normalize_weights(weights)
    # ratios should be preserved
    assert torch.allclose(out / out[0], weights / weights[0], atol=1e-5)
