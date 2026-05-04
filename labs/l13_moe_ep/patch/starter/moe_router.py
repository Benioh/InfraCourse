"""
L05.5 Patch · MoE Top-2 Router

填空规则：
- TODO(student) 必须自己写
- 不许 import tutel / fairscale.moe
- 允许 torch.softmax / topk / scatter

完成度自检：
    make patch-test M=l13_moe_ep
"""

from __future__ import annotations

from typing import Tuple

import torch


def top2_router(
    logits: torch.Tensor,
    capacity_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits: (num_tokens, num_experts).

    Returns:
        dispatch_mask:    (num_tokens, num_experts) float in {0, 1}
        combine_weights:  (num_tokens, num_experts) float (0 for non-selected; sums to <= 1 per row)
        aux_loss:         scalar tensor
    """
    num_tokens, num_experts = logits.shape

    # 1. softmax → 每个 token 在 num_experts 上的概率分布
    # TODO(student): probs = torch.softmax(logits, dim=-1)
    raise NotImplementedError("L05.5: implement softmax")

    # 2. top-2: 每行选概率最大的 2 个 expert（同时拿到对应权重）
    # TODO(student):
    #   top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)   # (num_tokens, 2)
    #   把 top2_vals 沿 last dim 重新归一化（因为只保留 2 个，要让 weights 之和 ≈ 1）
    #   normalized_vals = top2_vals / top2_vals.sum(-1, keepdim=True)

    # 3. 散列回 (num_tokens, num_experts) shape：
    #   combine_weights = torch.zeros_like(probs)
    #   combine_weights.scatter_(dim=-1, index=top2_idx, src=normalized_vals)
    #   dispatch_mask = (combine_weights > 0).float()

    # 4. Capacity 限制：每个 expert 最多接收 capacity 个 token
    #   capacity = int(capacity_factor * num_tokens * 2 / num_experts)
    #   对每列（每个 expert）：
    #     如果该 expert 收到的 token 数 > capacity，按 combine_weight 排序保留前 capacity 个，
    #     其余的 dispatch_mask 设为 0、combine_weights 设为 0
    #
    # 提示：
    #   for expert_id in range(num_experts):
    #     col = combine_weights[:, expert_id]
    #     count = (col > 0).sum()
    #     if count > capacity:
    #         # 找到前 capacity 大的 token 保留，其余清零
    #         _, top_token_idx = torch.topk(col, k=capacity)
    #         keep = torch.zeros_like(col, dtype=torch.bool)
    #         keep[top_token_idx] = True
    #         combine_weights[:, expert_id] = torch.where(keep, col, torch.zeros_like(col))
    #         dispatch_mask[:, expert_id] = keep.float()

    # 5. Aux loss（Switch Transformer 公式）
    #   fraction_routed = dispatch_mask.mean(dim=0)        # (num_experts,)
    #   fraction_prob   = probs.mean(dim=0)                # (num_experts,)
    #   aux_loss = num_experts * (fraction_routed * fraction_prob).sum()
    #
    # 注意：aux_loss 用 probs（softmax 输出）算 fraction_prob，而不是 normalized_vals。

    # TODO(student): 完成上面 5 步并 return (dispatch_mask, combine_weights, aux_loss)
    raise NotImplementedError("L05.5: implement top2_router body")
