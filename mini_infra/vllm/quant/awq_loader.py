"""
AWQ (Activation-aware Weight Quantization) 加载器骨架（L08.5 教学）。

教学目的：
    把 AWQ w4a16 的"per-group scale + zero point + packed int4"存储布局
    具体化。学员在 n17 notebook 里能：
        - 算出 hidden=4096, group_size=128 时有多少个 group
        - 看到压缩比（fp16 → int4 + per-group scale ≈ 4-5x）
        - 理解为什么 group_size 必须整除 hidden（不整除直接 raise）

真实框架对照：
    - github_repo/vllm/vllm/model_executor/layers/quantization/awq.py
        真实 AWQ 加载与 GEMM kernel dispatch；本文件只做布局与字节统计。
    - AutoAWQ 项目（github.com/casper-hansen/AutoAWQ）：AWQ 量化工具链。

简化掉的复杂度：
    - 不做真实 packed int4 → fp16 的 dequant kernel
    - scale 用启发式 1/(group_id+8)，不是真实校准结果
    - 不做 reorder / activation_order 优化
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AWQGroup:
    """AWQ 一组权重的元数据。

    每组 group_size 个权重共享一个 fp16 scale 和一个 int4 zero point。
    解包时 fp16_weight = (int4_packed - zero) × scale。
    """
    group_id: int
    start: int
    end: int
    scale: float
    zero_point: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def group_layout(hidden_size: int, group_size: int = 128) -> list[AWQGroup]:
    if hidden_size % group_size != 0:
        raise ValueError("AWQ group_size must divide hidden_size")
    groups = []
    for group_id, start in enumerate(range(0, hidden_size, group_size)):
        groups.append(
            AWQGroup(
                group_id=group_id,
                start=start,
                end=start + group_size,
                scale=round(1.0 / (group_id + 8), 6),
                zero_point=8,
            )
        )
    return groups


def awq_summary(
    hidden_size: int = 4096, group_size: int = 128, weight_bits: int = 4
) -> dict[str, object]:
    groups = group_layout(hidden_size, group_size)
    fp16_bytes = hidden_size * hidden_size * 2
    packed_bytes = hidden_size * hidden_size * weight_bits // 8
    scale_bytes = len(groups) * 2 * hidden_size
    return {
        "quant": "awq_w4a16",
        "hidden_size": hidden_size,
        "group_size": group_size,
        "num_groups": len(groups),
        "weight_bits": weight_bits,
        "fp16_weight_gb": round(fp16_bytes / 1e9, 6),
        "packed_weight_gb": round((packed_bytes + scale_bytes) / 1e9, 6),
        "compression_ratio": round(fp16_bytes / max(packed_bytes + scale_bytes, 1), 3),
        "groups": [group.to_dict() for group in groups[:4]],
    }
