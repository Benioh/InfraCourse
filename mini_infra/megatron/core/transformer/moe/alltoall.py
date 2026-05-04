"""
MoE All-to-All 通信代价模型（L05.5 教学）。

教学目的：
    教学员看清 MoE 一次 forward 的两次 all-to-all 是怎么来的、要发多少
    字节、与 EP size 的关系。结合 lab 跑真实 NCCL all-to-all，能解释
    为什么 EP=8 时 alltoall_ms 占 step 比例可达 30%+。

完整 MoE 一次 forward：
    1. permute (本地)：按 expert 桶整理 token
    2. all-to-all #1 (跨卡 dispatch)：把 token 发到对应 expert 所在 rank
    3. experts forward (本地，每 rank 跑分到自己的 expert)
    4. unpermute (本地)
    5. all-to-all #2 (跨卡 combine)：把 expert 输出送回原 token 所在 rank

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py
        AllToAllTokenDispatcher / AllGatherTokenDispatcher 两种 dispatch
        策略；本文件只算 AllToAll 的字节数与时延上界。

简化掉的复杂度：
    - 没有真实 NCCL；用 (bytes / bandwidth) 估算
    - bandwidth_gbs=300 是 H100 NVLink 量级的启发值
    - ring_factor 用 (ep-1)/ep 近似，真实 NCCL 有 tree/double-tree 等优化
"""
from __future__ import annotations


def permute_for_experts(routes: list[list[tuple[int, float]]], num_experts: int) -> list[list[int]]:
    """按 expert 把 token 分桶。

    输出 buckets[expert_id] = [token_id, ...]，对应 dispatch 前的本地
    permute 步骤。真实实现会同时输出 reverse 索引，用于 combine 后的
    unpermute。
    """
    buckets = [[] for _ in range(num_experts)]
    for token_id, token_routes in enumerate(routes):
        for expert_id, _ in token_routes:
            buckets[expert_id].append(token_id)
    return buckets


def alltoall_cost_ms(
    token_count: int, hidden_size: int, ep_size: int, bandwidth_gbs: float = 300.0
) -> float:
    payload_bytes = token_count * hidden_size * 2 * 2
    ring_factor = max(ep_size - 1, 1) / max(ep_size, 1)
    return round(payload_bytes * ring_factor / (bandwidth_gbs * 1e9) * 1000, 6)


def dispatch_plan(
    routes: list[list[tuple[int, float]]],
    num_experts: int,
    hidden_size: int = 4096,
    ep_size: int = 2,
) -> dict[str, object]:
    buckets = permute_for_experts(routes, num_experts)
    token_count = sum(len(bucket) for bucket in buckets)
    return {
        "pipeline": ["permute", "all_to_all", "experts", "unpermute", "all_to_all"],
        "tokens_per_expert": [len(bucket) for bucket in buckets],
        "alltoall_ms": alltoall_cost_ms(token_count, hidden_size, ep_size),
        "ep_size": ep_size,
    }
