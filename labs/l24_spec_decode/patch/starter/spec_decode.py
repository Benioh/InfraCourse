"""
L25 Patch · Greedy Speculative Decoding Verify

填空规则：
- TODO(student) 必须自己写
- 不许 import transformers
- 允许 torch / List

完成度自检：
    make patch-test M=l24_spec_decode
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def greedy_verify(
    draft_tokens: List[int],
    target_logits: torch.Tensor,
) -> Tuple[List[int], int]:
    """target_logits shape: (k+1, vocab)，对应 prompt + 0..k 个 draft token 之后的位置。

    Returns:
        accepted_tokens: List[int]  长度 = num_accepted + 1
        num_accepted: int  ∈ [0, len(draft_tokens)]
    """
    k = len(draft_tokens)
    # 提前算 target 在每个位置的 argmax
    target_argmax = target_logits.argmax(dim=-1).tolist()  # length k+1

    # TODO(student):
    #   accepted = []
    #   for i in range(k):
    #       if target_argmax[i] == draft_tokens[i]:
    #           accepted.append(draft_tokens[i])
    #       else:
    #           # 不接受：用 target argmax 当 bonus，立刻 return
    #           return accepted + [target_argmax[i]], len(accepted)
    #   # 全部接受：position k 的 argmax 是免费 bonus
    #   accepted.append(target_argmax[k])
    #   return accepted, k
    raise NotImplementedError("L25: implement greedy_verify")
