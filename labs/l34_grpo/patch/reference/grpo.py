"""Reference solution for L11.8 Patch."""

from __future__ import annotations

import torch


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < eps:
        return rewards - mean
    return (rewards - mean) / (std + eps)


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    total = rewards.sum()
    G = rewards.numel()
    baseline = (total - rewards) / (G - 1)
    return rewards - baseline


def grpo_loss(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict:
    mask = mask.to(log_probs.dtype)
    adv = advantages.unsqueeze(-1).expand_as(log_probs).to(log_probs.dtype)
    ratio = (log_probs - log_probs_old).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token_loss = -torch.minimum(unclipped, clipped)
    denom = mask.sum().clamp(min=1)
    policy_loss = (per_token_loss * mask).sum() / denom
    diff = (log_probs_ref - log_probs).clamp(-30, 30)
    kl = diff.exp() - diff - 1.0
    kl_loss = (kl * mask).sum() / denom
    total = policy_loss + kl_beta * kl_loss
    clipped_mask = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).to(log_probs.dtype)
    clipped_frac = (clipped_mask * mask).sum() / denom
    ratio_mean = (ratio * mask).sum() / denom
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "ratio_mean": ratio_mean,
        "ratio_clipped_frac": clipped_frac,
    }
