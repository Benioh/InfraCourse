"""
L29.5 Patch · Train-Infer Mismatch 修正算子组

填空规则：
- TODO(student) 必须自己写
- 不许 import torch.distributions.kl_divergence
- 允许 torch.exp / log / clamp / mean / where 等基础 op

完成度自检：
    make patch-test M=l29.5_train_infer_mismatch
"""

from __future__ import annotations

import math

import torch


def compute_k3_kl(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    """K3 KL（Schulman 估计器）：

        k3(x) = p(x)/q(x) - 1 - log(p(x)/q(x))

    输入是对数形式（已知 log_p 和 log_q），返回 mean over batch。
    数学性质：k3 ≥ 0，且 log_p == log_q 时严格为 0。
    """
    # TODO(student):
    #   log_ratio = logp_p - logp_q
    #   ratio = torch.exp(log_ratio)
    #   k3 = ratio - 1.0 - log_ratio
    #   return k3.mean()
    raise NotImplementedError("L29.5: implement compute_k3_kl")


def tis_correct(
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    advantages: torch.Tensor,
    lo: float = 0.5,
    hi: float = 2.0,
) -> torch.Tensor:
    """Truncated IS：把 ratio = exp(logp_new - logp_old) clamp 到 [lo, hi]，再乘 advantages。

    返回 shape 与 advantages 一致。
    """
    # TODO(student):
    #   ratio = torch.exp(logp_new - logp_old)
    #   ratio_clipped = torch.clamp(ratio, lo, hi)
    #   return ratio_clipped * advantages
    raise NotImplementedError("L29.5: implement tis_correct")


def mis_with_mask(
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    advantages: torch.Tensor,
    lo: float = 0.5,
    hi: float = 2.0,
) -> torch.Tensor:
    """Masked IS：如果 ratio 越界，直接 mask 为 0（梯度归零，不像 TIS 截到边界）。"""
    # TODO(student):
    #   ratio = torch.exp(logp_new - logp_old)
    #   mask = ((ratio >= lo) & (ratio <= hi)).to(ratio.dtype)
    #   return ratio * mask * advantages
    raise NotImplementedError("L29.5: implement mis_with_mask")


def geometric_seq_is(
    logp_old: torch.Tensor,  # (B, T)
    logp_new: torch.Tensor,  # (B, T)
    seq_lens: torch.Tensor,  # (B,) int64 token 数
) -> torch.Tensor:
    """序列级几何均值 IS：

        w_seq = exp((1/|y|) * Σ log(π_new/π_old))

    长度归一化让长短序列权重幅度可比。返回 (B,)。
    """
    # TODO(student):
    #   log_ratios = logp_new - logp_old   # (B, T)
    #   B, T = log_ratios.shape
    #   mask = (torch.arange(T, device=log_ratios.device)[None, :] < seq_lens[:, None]).to(log_ratios.dtype)
    #   sum_log = (log_ratios * mask).sum(dim=-1)
    #   denom = seq_lens.to(log_ratios.dtype).clamp(min=1.0)
    #   return torch.exp(sum_log / denom)
    raise NotImplementedError("L29.5: implement geometric_seq_is")


def apply_veto(logp_rollout: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Veto mask：对极端低概率 token 返回 0（直接 drop），否则 1。

    用 log 比较避免 exp 下溢：logp >= log(threshold) 即保留。
    """
    # TODO(student):
    #   log_thresh = math.log(threshold)
    #   return (logp_rollout >= log_thresh).to(logp_rollout.dtype)
    raise NotImplementedError("L29.5: implement apply_veto")


def batch_normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """SNIS：把整个 batch 的权重除以均值，让均值 = 1，避免有效学习率震荡。"""
    # TODO(student):
    #   mean = weights.mean()
    #   return weights / mean.clamp(min=1e-12)
    raise NotImplementedError("L29.5: implement batch_normalize_weights")
