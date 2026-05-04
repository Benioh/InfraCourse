"""L10.3 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.dpo")


def test_compute_logps_shape():
    impl = _impl()
    logits = torch.randn(3, 5, 7)
    labels = torch.randint(0, 7, (3, 5))
    out = impl.compute_logps_for_completions(logits, labels)
    assert out.shape == (3,)


def test_compute_logps_masks_prompt():
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels_full = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
    labels_masked = labels_full.clone()
    labels_masked[:, :2] = -100
    full = impl.compute_logps_for_completions(logits, labels_full)
    masked = impl.compute_logps_for_completions(logits, labels_masked)
    # log-probs are negative; masking out the first two tokens drops two negative
    # contributions, so masked sum must be GREATER (closer to zero) than the full sum.
    assert (masked > full).all()


def test_compute_logps_matches_manual():
    impl = _impl()
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[1, -100, 3, 4], [-100, 1, 2, -100]])
    out = impl.compute_logps_for_completions(logits, labels)
    log_probs = F.log_softmax(logits, dim=-1)
    safe = labels.clone()
    safe[labels == -100] = 0
    gathered = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    expected = (gathered * (labels != -100)).sum(dim=-1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dpo_zero_beta_returns_log2():
    impl = _impl()
    out = impl.dpo_loss(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        beta=0.0,
    )
    assert torch.isclose(out["loss"], torch.tensor(math.log(2.0)), atol=1e-6)


def test_dpo_policy_equals_ref_returns_log2():
    impl = _impl()
    chosen = torch.tensor([1.0, 2.0, -3.0])
    rejected = torch.tensor([0.5, -1.0, 0.7])
    out = impl.dpo_loss(chosen, rejected, chosen, rejected, beta=0.5)
    assert torch.isclose(out["loss"], torch.tensor(math.log(2.0)), atol=1e-6)


def test_dpo_chosen_better_lowers_loss():
    impl = _impl()
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    base = impl.dpo_loss(
        torch.tensor([0.0]), torch.tensor([0.0]), ref_chosen, ref_rejected, beta=0.5
    )["loss"]
    better = impl.dpo_loss(
        torch.tensor([1.0]), torch.tensor([-1.0]), ref_chosen, ref_rejected, beta=0.5
    )["loss"]
    assert better < base


def test_dpo_rejected_better_raises_loss():
    impl = _impl()
    out = impl.dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=0.5,
    )
    assert out["loss"] > math.log(2.0)


def test_dpo_reward_margin_formula():
    impl = _impl()
    pc = torch.tensor([1.0, 2.0])
    pr = torch.tensor([0.5, -1.0])
    rc = torch.tensor([0.0, 0.5])
    rr = torch.tensor([0.5, 0.0])
    beta = 0.4
    out = impl.dpo_loss(pc, pr, rc, rr, beta=beta)
    expected_margin = beta * ((pc - rc) - (pr - rr))
    assert torch.allclose(out["reward_margin"], expected_margin, atol=1e-6)


def test_dpo_numerically_stable_with_large_beta():
    impl = _impl()
    out = impl.dpo_loss(
        torch.tensor([10.0]),
        torch.tensor([-10.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=20.0,
    )
    assert torch.isfinite(out["loss"])
