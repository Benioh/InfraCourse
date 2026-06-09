"""
MoE Router 最小同构（L14 教学）。

教学目的：
    把 "router 决定 token 去哪个 expert" 从概念变成可断言的数学。
    我们提供 top-k softmax router、auxiliary load-balancing loss、
    可控的 collapse 模拟，让学员能在 CPU 上观察 expert imbalance、
    aux loss 系数对 router_entropy 的影响，再去看真实 Megatron MoE。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/transformer/moe/router.py
        TopKRouter / Sinkhorn / SwitchRouter 的真实实现；本文件保留
        TopK + softmax + aux_loss 的最简骨架。
    - github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py
        AllGather / AllToAll dispatcher 的工程化分支；本文件不实现 dispatch。

简化掉的复杂度：
    - 没有 NCCL all-to-all，只算 token 直方图
    - 没有 jitter / noise，collapse 用人工 bias 触发
    - aux_loss 用方差度量，真实实现常用 entropy 或 token fraction × prob
"""
from __future__ import annotations

import math
from collections import Counter


def softmax(values: list[float]) -> list[float]:
    """数值稳定 softmax，用于把 router logits 转成概率。"""

    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def route_tokens(logits: list[list[float]], top_k: int = 2) -> list[list[tuple[int, float]]]:
    """对每个 token 做 top-k softmax，返回 [(expert_id, prob)] 列表。

    真实 Megatron 用 `torch.topk(probs, k)` 做相同事，并把这些 (expert, prob)
    传给 token_dispatcher 做 permute → all-to-all。本文件只到 routing 表，
    不做实际 dispatch。
    """

    routes = []
    for row in logits:
        probs = softmax(row)
        selected = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:top_k]
        routes.append([(expert, round(prob, 6)) for expert, prob in selected])
    return routes


def tokens_per_expert(routes: list[list[tuple[int, float]]], num_experts: int) -> list[int]:
    counts = Counter(expert for token_routes in routes for expert, _ in token_routes)
    return [counts.get(expert, 0) for expert in range(num_experts)]


def router_entropy(routes: list[list[tuple[int, float]]], num_experts: int) -> float:
    counts = tokens_per_expert(routes, num_experts)
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    return round(entropy / max(math.log(num_experts), 1e-9), 6)


def auxiliary_load_balance_loss(routes: list[list[tuple[int, float]]], num_experts: int) -> float:
    """方差形式的 aux loss。

    真实 Megatron 用 `mean(token_fraction * mean_prob) * num_experts`，
    本文件用 `var(token_count)` 让数学最直观：方差越大说明分布越不均，
    aux loss 越大。aux_loss 系数从 0 升到 0.01 一般能阻止 collapse。
    """

    counts = tokens_per_expert(routes, num_experts)
    total = max(sum(counts), 1)
    expected = total / num_experts
    return round(sum((count - expected) ** 2 for count in counts) / (num_experts * total**2), 8)


def synthetic_logits(
    num_tokens: int = 64, num_experts: int = 8, collapse: bool = False
) -> list[list[float]]:
    """生成 router 输入 logits。

    `collapse=True` 时把 expert 0 的 logit + 4，模拟训练初期某 expert
    略占优、最终垄断所有 token 的退化路径。学员可通过观察
    tokens_per_expert / router_entropy 复现 collapse，再加 aux_loss 修复。
    """

    logits = []
    for token in range(num_tokens):
        row = [math.sin((token + 1) * (expert + 1)) for expert in range(num_experts)]
        if collapse:
            row[0] += 4.0
        logits.append(row)
    return logits
