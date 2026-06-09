"""
MoE 端到端聚合（L14 教学入口）。

教学目的：
    把 router → capacity → all-to-all 三段拼成可一键调用的 moe_summary，
    学员跑一次就能拿到一组完整指标：
        active_params / total_params / router_entropy / aux_loss /
        capacity_overflow_rate / tokens_per_expert_p50/p99 / alltoall_ms /
        step_time_ms
    再去 lab 真实 Megatron MoE run 验证。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/transformer/moe/experts.py
        真实 expert 是 SwiGLU MLP；本文件不做 forward，只算指标。
    - github_repo/Megatron-LM/megatron/core/transformer/moe/moe_layer.py
        真实 MoE layer 把以上几段串起来。

简化掉的复杂度：
    - 不做实际 expert forward / backward
    - active_params 计算仅按比例估算
    - step_time_ms 用启发式公式给出量级
"""
from __future__ import annotations

from .alltoall import dispatch_plan
from .capacity import apply_capacity
from .router import auxiliary_load_balance_loss, route_tokens, router_entropy, synthetic_logits


def moe_summary(
    num_tokens: int = 64,
    num_experts: int = 8,
    top_k: int = 2,
    capacity_factor: float = 1.25,
    ep_size: int = 2,
    collapse: bool = False,
) -> dict[str, object]:
    logits = synthetic_logits(num_tokens, num_experts, collapse=collapse)
    routes = route_tokens(logits, top_k=top_k)
    capacity = apply_capacity(routes, num_experts, capacity_factor)
    dispatch = dispatch_plan(routes, num_experts, ep_size=ep_size)
    active_params = num_experts * top_k * 4096 * 4096 // max(num_experts, 1)
    total_params = num_experts * 4096 * 4096
    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "capacity_factor": capacity_factor,
        "active_params": active_params,
        "total_params": total_params,
        "router_entropy": router_entropy(routes, num_experts),
        "aux_loss": auxiliary_load_balance_loss(routes, num_experts),
        "capacity_overflow_rate": capacity["capacity_overflow_rate"],
        "tokens_per_expert_p50": sorted(dispatch["tokens_per_expert"])[num_experts // 2],
        "tokens_per_expert_p99": max(dispatch["tokens_per_expert"]),
        "alltoall_ms": dispatch["alltoall_ms"],
        "step_time_ms": round(8.0 + float(dispatch["alltoall_ms"]) + num_tokens * 0.02, 6),
        "dispatch": dispatch,
        "capacity": capacity,
    }
