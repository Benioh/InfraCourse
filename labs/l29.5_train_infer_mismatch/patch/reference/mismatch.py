"""Reference solution for L29.5 Patch · Train-Infer Mismatch."""

from __future__ import annotations

import math

import torch


def compute_k3_kl(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    log_ratio = logp_p - logp_q
    ratio = torch.exp(log_ratio)
    k3 = ratio - 1.0 - log_ratio
    return k3.mean()


def tis_correct(
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    advantages: torch.Tensor,
    lo: float = 0.5,
    hi: float = 2.0,
) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_old)
    ratio_clipped = torch.clamp(ratio, lo, hi)
    return ratio_clipped * advantages


def mis_with_mask(
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    advantages: torch.Tensor,
    lo: float = 0.5,
    hi: float = 2.0,
) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_old)
    mask = ((ratio >= lo) & (ratio <= hi)).to(ratio.dtype)
    return ratio * mask * advantages


def geometric_seq_is(
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    log_ratios = logp_new - logp_old
    B, T = log_ratios.shape
    arange = torch.arange(T, device=log_ratios.device)
    mask = (arange[None, :] < seq_lens[:, None]).to(log_ratios.dtype)
    sum_log = (log_ratios * mask).sum(dim=-1)
    denom = seq_lens.to(log_ratios.dtype).clamp(min=1.0)
    return torch.exp(sum_log / denom)


def apply_veto(logp_rollout: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    log_thresh = math.log(threshold)
    return (logp_rollout >= log_thresh).to(logp_rollout.dtype)


def batch_normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    mean = weights.mean()
    return weights / mean.clamp(min=1e-12)
