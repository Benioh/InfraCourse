"""Reference solution for L20 Patch · typical_p_filter."""

from __future__ import annotations

import torch


def typical_p_filter(
    logits: torch.Tensor,
    typical_p: float = 0.9,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    if typical_p >= 1.0:
        return logits

    probs = torch.softmax(logits.float(), dim=-1)
    info = -torch.log(probs + 1e-10)
    entropy = (probs * info).sum(dim=-1, keepdim=True)
    dist = (info - entropy).abs()

    _, sorted_idx = torch.sort(dist, dim=-1)
    sorted_probs = probs.gather(-1, sorted_idx)
    cumsum = sorted_probs.cumsum(dim=-1)

    # token at position k is removed if cumsum[k] > typical_p
    # but we want at least 1 kept → shift right by 1: keep if PREVIOUS cumsum <= typical_p
    sorted_remove = cumsum > typical_p
    # The first position must always be kept (closest to typical surprisal)
    sorted_remove[..., 0] = False

    mask_to_remove = torch.zeros_like(sorted_remove)
    mask_to_remove.scatter_(-1, sorted_idx, sorted_remove)

    return logits.masked_fill(mask_to_remove, filter_value)
