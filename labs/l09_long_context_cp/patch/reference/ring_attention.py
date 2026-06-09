"""Reference solution for L10 Patch · ring attention forward."""

from __future__ import annotations

import math

import torch


def ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: int = 1,
) -> torch.Tensor:
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    k_chunks = list(torch.chunk(k, num_chunks, dim=2))
    v_chunks = list(torch.chunk(v, num_chunks, dim=2))

    running_out = torch.zeros(B, H, Sq, D, dtype=q.dtype, device=q.device)
    running_max = torch.full(
        (B, H, Sq, 1), float("-inf"), dtype=q.dtype, device=q.device
    )
    running_denom = torch.zeros(B, H, Sq, 1, dtype=q.dtype, device=q.device)

    for k_chunk, v_chunk in zip(k_chunks, v_chunks):
        scores = torch.einsum("bhid,bhjd->bhij", q, k_chunk) * scale
        chunk_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        # First iteration: running_max is -inf, so exp(-inf - new_max) = 0; safe under
        # the convention used here because we initialize running_out / running_denom to 0.
        exp_old = torch.where(
            torch.isfinite(running_max),
            torch.exp(running_max - new_max),
            torch.zeros_like(new_max),
        )
        exp_chunk = torch.exp(scores - new_max)
        running_denom = running_denom * exp_old + exp_chunk.sum(dim=-1, keepdim=True)
        running_out = running_out * exp_old + torch.einsum(
            "bhij,bhjd->bhid", exp_chunk, v_chunk
        )
        running_max = new_max

    return running_out / running_denom
