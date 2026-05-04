"""
MoE Capacity Factor 与 dropping 语义（L05.5 教学）。

教学目的：
    把 "每个 expert 最多能接 (tokens/experts × cf) 个 token，超出就 drop"
    从经验话变成可断言的数学。学员能在 n15 notebook 中看到：
        - cf=1.0 时，imbalance 一旦发生立刻有 token 被 drop
        - cf=2.0 时，drop 几乎为 0，但每个 expert 显存占用翻倍
        - capacity_overflow_rate 与 router collapse 强相关
    再回到 lab 用 aux_loss 与 cf 联调。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/transformer/moe/router.py
        capacity 计算与 token drop/reroute 的完整逻辑；本文件只做基本
        drop（不做 reroute）。
    - GShard 论文（Lepikhin et al. 2020）首先引入 capacity factor 概念。

简化掉的复杂度：
    - 不做 reroute（真实有的实现会把 dropped token 送给次优 expert）
    - 不做 padding（真实 dispatcher 需要把每个 expert 的输入 pad 到 capacity）
    - 不与 backward 联动（真实 dropping 在 backward 时会被 mask 掉）
"""
from __future__ import annotations

import math


def capacity_per_expert(
    num_tokens: int, num_experts: int, top_k: int, capacity_factor: float
) -> int:
    """计算每个 expert 的 capacity 上限。

    公式：capacity = ceil(tokens × top_k / experts × cf)
    cf=1.0 即"每个 expert 平均承担 tokens/experts 个 token，无冗余"。
    cf=1.25 是 GShard 默认；cf=2.0 给容量但翻倍显存。
    """
    raw = num_tokens * top_k / max(num_experts, 1) * capacity_factor
    return max(math.ceil(raw), 1)


def apply_capacity(
    routes: list[list[tuple[int, float]]],
    num_experts: int,
    capacity_factor: float = 1.25,
) -> dict[str, object]:
    top_k = len(routes[0]) if routes else 1
    capacity = capacity_per_expert(len(routes), num_experts, top_k, capacity_factor)
    used = [0 for _ in range(num_experts)]
    kept = 0
    dropped = 0
    decisions = []
    for token_id, token_routes in enumerate(routes):
        token_decisions = []
        for expert_id, probability in token_routes:
            accepted = used[expert_id] < capacity
            if accepted:
                used[expert_id] += 1
                kept += 1
            else:
                dropped += 1
            token_decisions.append(
                {
                    "token_id": token_id,
                    "expert_id": expert_id,
                    "probability": probability,
                    "accepted": accepted,
                }
            )
        decisions.extend(token_decisions)
    total = max(kept + dropped, 1)
    return {
        "capacity_per_expert": capacity,
        "tokens_per_expert_after_capacity": used,
        "kept": kept,
        "dropped": dropped,
        "capacity_overflow_rate": round(dropped / total, 6),
        "decisions": decisions,
    }
