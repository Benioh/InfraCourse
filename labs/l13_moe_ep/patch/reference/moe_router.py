"""Reference solution for L14 Patch · top2_router."""

from __future__ import annotations

from typing import Tuple

import torch


def top2_router(
    logits: torch.Tensor,
    capacity_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = logits.shape

    probs = torch.softmax(logits, dim=-1)
    top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)
    normalized = top2_vals / (top2_vals.sum(dim=-1, keepdim=True) + 1e-9)

    combine_weights = torch.zeros_like(probs)
    combine_weights.scatter_(dim=-1, index=top2_idx, src=normalized)
    dispatch_mask = (combine_weights > 0).float()

    capacity = max(1, int(capacity_factor * num_tokens * 2 / num_experts))
    for e in range(num_experts):
        col = combine_weights[:, e]
        count = int((col > 0).sum().item())
        if count > capacity:
            _, top_idx = torch.topk(col, k=capacity)
            keep = torch.zeros_like(col, dtype=torch.bool)
            keep[top_idx] = True
            combine_weights[:, e] = torch.where(keep, col, torch.zeros_like(col))
            dispatch_mask[:, e] = keep.float()

    fraction_routed = dispatch_mask.mean(dim=0)
    fraction_prob = probs.mean(dim=0)
    aux_loss = num_experts * (fraction_routed * fraction_prob).sum()

    return dispatch_mask, combine_weights, aux_loss
