"""L11.8 Patch · GRPO / RLOO advantage + loss."""

from __future__ import annotations

import torch


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative mean-0 std-1 advantage."""
    # TODO(student): if rewards.numel() <= 1, return zeros_like(rewards)
    # TODO(student): mean = rewards.mean(); std = rewards.std(unbiased=False)
    # TODO(student): if std < eps return rewards - mean (or zeros)
    # TODO(student): return (rewards - mean) / (std + eps)
    raise NotImplementedError("L11.8: implement grpo_advantage")


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline."""
    # TODO(student): if rewards.numel() <= 1, return zeros_like(rewards)
    # TODO(student): total = rewards.sum()
    # TODO(student): baseline_i = (total - rewards) / (G - 1)
    # TODO(student): return rewards - baseline_i
    raise NotImplementedError("L11.8: implement rloo_advantage")


def grpo_loss(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict:
    # TODO(student): broadcast advantages [G] -> [G, T]
    # TODO(student): ratio = (log_probs - log_probs_old).exp()
    # TODO(student): unclipped = ratio * advantages_expanded
    # TODO(student): clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages_expanded
    # TODO(student): policy_loss = -torch.minimum(unclipped, clipped) (per-token, multiplied by mask, normalized by mask.sum())
    # TODO(student): kl = (log_probs_ref - log_probs).exp() - (log_probs_ref - log_probs) - 1
    # TODO(student): kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)
    # TODO(student): total = policy_loss + kl_beta * kl_loss
    # TODO(student): also report ratio_mean and clipped_frac
    raise NotImplementedError("L11.8: implement grpo_loss")
