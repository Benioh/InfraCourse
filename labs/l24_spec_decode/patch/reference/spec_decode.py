"""Reference solution for L25 Patch · greedy_verify."""

from __future__ import annotations

from typing import List, Tuple

import torch


def greedy_verify(
    draft_tokens: List[int],
    target_logits: torch.Tensor,
) -> Tuple[List[int], int]:
    k = len(draft_tokens)
    target_argmax = target_logits.argmax(dim=-1).tolist()
    accepted: List[int] = []
    for i in range(k):
        if target_argmax[i] == draft_tokens[i]:
            accepted.append(draft_tokens[i])
        else:
            return accepted + [target_argmax[i]], len(accepted)
    accepted.append(target_argmax[k])
    return accepted, k
