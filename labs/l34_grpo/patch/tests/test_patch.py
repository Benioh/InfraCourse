"""L39 Patch tests · CPU only."""

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
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.grpo")


def test_group_advantage_zero_mean():
    impl = _impl()
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    adv = impl.grpo_advantage(rewards)
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-5)


def test_group_advantage_unit_std():
    impl = _impl()
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    adv = impl.grpo_advantage(rewards)
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-3)


def test_single_response_group_returns_zero():
    impl = _impl()
    rewards = torch.tensor([3.14])
    adv = impl.grpo_advantage(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_advantage_constant_rewards_zero():
    impl = _impl()
    rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])
    adv = impl.grpo_advantage(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)


def test_rloo_advantage_uses_other_responses():
    impl = _impl()
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = impl.rloo_advantage(r)
    # A_0 = 1 - mean(2,3,4) = 1 - 3 = -2
    assert torch.isclose(adv[0], torch.tensor(-2.0), atol=1e-6)
    # A_3 = 4 - mean(1,2,3) = 4 - 2 = 2
    assert torch.isclose(adv[3], torch.tensor(2.0), atol=1e-6)


def test_rloo_zero_when_all_equal():
    impl = _impl()
    r = torch.tensor([5.0, 5.0, 5.0])
    adv = impl.rloo_advantage(r)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_grpo_loss_with_kl_penalty():
    impl = _impl()
    G, T = 4, 6
    torch.manual_seed(0)
    lp = torch.randn(G, T)
    lp_old = lp.clone()
    lp_ref = lp + 0.5  # nontrivial KL
    advantages = impl.grpo_advantage(torch.tensor([1.0, 0.0, 2.0, -1.0]))
    mask = torch.ones(G, T)
    out_with_kl = impl.grpo_loss(lp, lp_old, lp_ref, advantages, mask, clip_eps=0.2, kl_beta=0.1)
    out_no_kl = impl.grpo_loss(lp, lp_old, lp_ref, advantages, mask, clip_eps=0.2, kl_beta=0.0)
    assert out_with_kl["kl_loss"] > 0
    assert not torch.isclose(out_with_kl["loss"], out_no_kl["loss"])


def test_grpo_loss_clip_caps_ratio():
    impl = _impl()
    G, T = 2, 3
    lp = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    lp_old = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    lp_ref = lp.clone()
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.ones(G, T)
    out = impl.grpo_loss(lp, lp_old, lp_ref, advantages, mask, clip_eps=0.2, kl_beta=0.0)
    # ratio for row 0 = e^1 ≈ 2.718; should be clipped fraction > 0
    assert out["ratio_clipped_frac"] > 0


def test_grpo_loss_per_token_mask():
    impl = _impl()
    G, T = 2, 4
    torch.manual_seed(0)
    lp = torch.randn(G, T)
    lp_old = lp.clone()
    lp_ref = lp.clone()
    advantages = torch.tensor([1.0, -1.0])
    mask_full = torch.ones(G, T)
    mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    out_full = impl.grpo_loss(lp, lp_old, lp_ref, advantages, mask_full, kl_beta=0.0)
    out_half = impl.grpo_loss(lp, lp_old, lp_ref, advantages, mask_half, kl_beta=0.0)
    assert not torch.isclose(out_full["loss"], out_half["loss"])
