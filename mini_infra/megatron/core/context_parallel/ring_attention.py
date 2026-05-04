"""
Ring Attention 最小同构骨架（L04.5 教学）。

教学目的：
    把 "Q stationary + K/V 在 ring 上滚动 + 每步累加 partial softmax"
    从论文公式变成可枚举的 RingStep 列表。学员能直接在 CPU 上看出：
        - CP=4 一个 step 共 4×3=12 次 K/V 传递
        - 每次传递的字节数 = chunk × hidden × dtype × 2 (K + V)
        - 总通信量与 cp_size×(cp_size-1) 成正比
    再结合 mini_infra/gpu/memory_model 的带宽规格，就能预测 attn_comm_bytes
    在 step_time 中的占比。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/transformer/attention.py
        真实 ring attention：partial softmax 累加用 (max, sum) 重缩放，
        且 causal mask 在 CP 下需重新切分。本文件只算拓扑与字节，不做
        真实 attention 计算。
    - github_repo/Megatron-LM/megatron/core/parallel_state.py
        CP group 创建（model_parallel_group / context_parallel_group）。

简化掉的复杂度：
    - 没有真实 partial softmax 累加（学员去 n14 notebook 用 numpy 验证）
    - 没有 causal mask 切分逻辑
    - bytes_per_chunk 假设 K 与 V 一起发，真实可能 fuse 进一次 sendrecv
    - step_time_ms 公式（seq_len/128 + cp*8 + bytes/1e8）是教学启发式，
      非真实硬件实测
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RingStep:
    """ring 上一次 K/V 传递事件。

    q_rank: Q 留在哪个 rank（stationary）
    kv_rank: 这一步从哪个 rank 拿来 K/V chunk
    seq_start/end: 这次 K/V chunk 对应的 seq 切片
    bytes_sent: 本次跨卡通信的字节数（K + V）
    """
    step: int
    q_rank: int
    kv_rank: int
    seq_start: int
    seq_end: int
    bytes_sent: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def attention_memory_gb(seq_len: int, hidden_size: int, dtype_bytes: int = 2) -> float:
    qkv = 3 * seq_len * hidden_size * dtype_bytes
    logits = seq_len * seq_len * dtype_bytes
    return round((qkv + logits) / 1e9, 4)


def ring_attention_plan(
    seq_len: int = 16384,
    cp_size: int = 2,
    hidden_size: int = 4096,
    dtype_bytes: int = 2,
) -> dict[str, object]:
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    chunk = seq_len // cp_size
    bytes_per_chunk = chunk * hidden_size * dtype_bytes * 2
    steps = []
    for step in range(cp_size):
        for q_rank in range(cp_size):
            kv_rank = (q_rank - step) % cp_size
            steps.append(
                RingStep(
                    step=step,
                    q_rank=q_rank,
                    kv_rank=kv_rank,
                    seq_start=kv_rank * chunk,
                    seq_end=(kv_rank + 1) * chunk,
                    bytes_sent=bytes_per_chunk,
                ).to_dict()
            )
    return {
        "seq_len": seq_len,
        "cp_size": cp_size,
        "hidden_size": hidden_size,
        "q_stationary": True,
        "kv_ring_steps": cp_size,
        "attn_comm_bytes": bytes_per_chunk * cp_size * max(cp_size - 1, 0),
        "peak_mem_gb_cp": round(
            attention_memory_gb(seq_len, hidden_size, dtype_bytes) / cp_size, 4
        ),
        "steps": steps,
    }


def seqlen_sweep() -> list[dict[str, object]]:
    rows = []
    for seq_len in (4096, 16384, 65536):
        for cp_size in (1, 2, 4):
            plan = ring_attention_plan(seq_len=seq_len, cp_size=cp_size)
            plan.update(
                {
                    "step_time_ms": round(
                        seq_len / 128 + cp_size * 8 + plan["attn_comm_bytes"] / 1e8, 3
                    ),
                    "tokens_per_sec": round(seq_len / max(seq_len / 128 + cp_size * 8, 1), 3),
                }
            )
            rows.append(plan)
    return rows
