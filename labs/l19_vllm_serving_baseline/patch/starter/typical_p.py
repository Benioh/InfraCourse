"""
L07 Patch · Typical_p sampling

填空规则：
- TODO(student) 必须自己写
- 不许 import transformers
- 允许 torch.softmax / log / sort / scatter

完成度自检：
    make patch-test M=l19_vllm_serving_baseline
"""

from __future__ import annotations

import torch


def typical_p_filter(
    logits: torch.Tensor,
    typical_p: float = 0.9,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """logits: (batch, vocab). 返回过滤后的 logits（同形）。"""
    if typical_p >= 1.0:
        return logits

    # TODO(student):
    # 1. probs = torch.softmax(logits.float(), dim=-1)
    # 2. info = -torch.log(probs + 1e-10)              # (B, V)
    # 3. entropy = (probs * info).sum(dim=-1, keepdim=True)  # (B, 1)
    # 4. dist = (info - entropy).abs()                  # (B, V)
    #
    # 5. 按 dist 升序排序得到顺序索引
    #    sorted_dist, sorted_idx = torch.sort(dist, dim=-1)
    #    sorted_probs = probs.gather(-1, sorted_idx)
    #    cumsum = sorted_probs.cumsum(dim=-1)            # 累加 mass
    #
    # 6. 找到 cumsum > typical_p 的位置；该位置（含）之后的 token 全 mask
    #    sorted_remove = cumsum > typical_p
    #    # 至少保留 1 个：把第 0 列强制 False
    #    sorted_remove[..., 0] = False
    #
    # 7. 把 sorted_remove scatter 回原始 vocab 顺序得到 mask_to_remove
    #    mask_to_remove = torch.zeros_like(sorted_remove)
    #    mask_to_remove.scatter_(-1, sorted_idx, sorted_remove)
    #
    # 8. result = logits.masked_fill(mask_to_remove, filter_value)
    #    return result
    raise NotImplementedError("L07 Patch: implement typical_p_filter")
