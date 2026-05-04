"""
Speculative decoding draft + verify 骨架（L08.7 教学）。

教学目的：
    把投机解码一次 step 的"draft 提议 K → target 一次 forward 验证 →
    accept 前 j 个 → KV 回滚后 K-j 个"具体化。学员能看到：
        - acceptance < 1 时为什么仍可能加速（一次 forward 能跑过 j+1 token）
        - acceptance 与 draft_cost_ratio 共同决定 speedup
        - OOD prompt 时 acceptance 会塌（domain="out_of_domain"）
        - 高并发时 itl_p99 退化（公式中 1.8x at concurrency≥64）

真实框架对照：
    - github_repo/vllm/vllm/spec_decode/spec_decode_worker.py
        真实 draft + target 的协同、KV 共享与回滚
    - github_repo/vllm/vllm/spec_decode/metrics.py
        acceptance rate 的工程化采集

简化掉的复杂度：
    - 没有真实 draft model；future tokens 直接用预设串
    - speedup 公式是教学用近似（1 / ((1-rate) + draft_cost_ratio)）
    - 不实现 KV 回滚（真实需要 target 在 verify 后恢复 KV 状态）
"""
from __future__ import annotations

from .acceptance_tracker import AcceptanceTracker
from .ngram import ngram_candidates


def verify_candidates(target_tokens: list[str], candidates: list[str]) -> int:
    """target 一次 forward 验证 K 个候选，返回连续 accept 的数量。

    关键不变量：accept 失败的位置之后所有候选都拒，不能"挑着 accept"。
    真实实现中 target forward 的 logits 用作"金标准"，第一处不一致就停。
    """
    accepted = 0
    for target, candidate in zip(target_tokens, candidates, strict=False):
        if target != candidate:
            break
        accepted += 1
    return accepted


def draft_step(
    prompt_tokens: list[str], target_future: list[str], k: int = 4, mode: str = "ngram"
) -> dict[str, object]:
    if mode == "ngram":
        candidates = ngram_candidates(prompt_tokens, prompt_tokens[-3:], max_tokens=k)
    else:
        candidates = target_future[: max(k - 1, 0)] + ["<miss>"]
    accepted = verify_candidates(target_future, candidates)
    return {
        "mode": mode,
        "proposed": len(candidates),
        "accepted": accepted,
        "kv_rollback_tokens": max(len(candidates) - accepted, 0),
        "candidates": candidates,
    }


def spec_decode_summary(
    mode: str = "ngram", concurrency: int = 1, domain: str = "in_domain"
) -> dict[str, object]:
    prompt = "solve math step by step solve math step by step".split()
    future = (
        "solve math step by step".split()
        if domain == "in_domain"
        else "write poetry about kernels".split()
    )
    tracker = AcceptanceTracker(window=4)
    step = draft_step(prompt, future, mode=mode)
    tracker.add(int(step["accepted"]), max(int(step["proposed"]), 1))
    base_itl = 18.0 + concurrency * 0.2
    speedup = tracker.expected_speedup(draft_cost_ratio=0.2 if mode == "ngram" else 0.35)
    return {
        **step,
        "acceptance_rate": tracker.rate(),
        "speedup": speedup,
        "ttft_ms": round(120 + concurrency * 1.5, 3),
        "itl_ms_p50": round(base_itl / max(speedup, 1e-6), 3),
        "itl_ms_p99": round(
            base_itl * (1.8 if concurrency >= 64 else 1.25) / max(speedup, 1e-6), 3
        ),
        "draft_latency_ms": 0.0 if mode == "ngram" else 6.0,
        "draft_gpu_mem_gb": 0.0 if mode == "ngram" else 2.4,
        "domain": domain,
    }
