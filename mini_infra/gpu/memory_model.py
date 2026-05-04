"""
GPU 内存层级 + Roofline 估算（L01.7 教学骨架）。

教学目的：
    在跑真实 Triton kernel 之前，先建立"性能上界来自哪里"的预算思维。
    我们不实测带宽，只用规格表里的 peak HBM 带宽与 SM 资源限额，估算：
        - fused softmax 应该读写多少字节（vs eager softmax 多读 ~2.5x）
        - 给定 BLOCK_SIZE / num_warps 时 occupancy 大约是多少
        - 这些组合下 bandwidth_gbs 会落在 peak 的什么比例

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/fusions/fused_softmax.py：生产 fused
      kernel 的 backend dispatch（ScaledSoftmax / ScaledMaskedSoftmax 等）
    - github_repo/triton/python/tutorials/02-fused-softmax.py：autotune 在真实
      硬件上的搜索空间

简化掉的复杂度：
    - 没考虑 L2 cache 命中（实测中 L2 命中可让有效带宽超过 HBM peak）
    - 没考虑 register spill 对 occupancy 的影响
    - 启发式系数（如 block_efficiency 中的 0.55/4096）只用于教学，给出量级
      正确但不精确的预算；真实硬件请用 ncu/kineto 覆盖
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUProfile:
    """单张 GPU 的关键资源与带宽规格。

    字段对应 Nsight Compute 的 device 信息：
        peak_hbm_gbs: HBM 带宽上界（GB/s），决定 memory-bound kernel 的天花板。
        l2_mb: L2 cache 容量；本课程中不参与估算，仅供学员对照。
        smem_kb_per_sm: 单个 SM 的 shared memory 上限，BLOCK_SIZE 太大会溢出。
        registers_kb_per_sm: 单个 SM 的寄存器总量；spill 会降低 occupancy。
        max_warps_per_sm: occupancy 的分母，4090 与 H200 都是 64。
    """

    name: str
    peak_hbm_gbs: float
    l2_mb: float
    smem_kb_per_sm: float
    registers_kb_per_sm: float
    max_warps_per_sm: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


GPU_PROFILES: dict[str, GPUProfile] = {
    "cpu-validation": GPUProfile(
        name="cpu-validation",
        peak_hbm_gbs=80.0,
        l2_mb=16.0,
        smem_kb_per_sm=0.0,
        registers_kb_per_sm=0.0,
        max_warps_per_sm=1,
    ),
    "rtx4090": GPUProfile(
        name="rtx4090",
        peak_hbm_gbs=1008.0,
        l2_mb=72.0,
        smem_kb_per_sm=100.0,
        registers_kb_per_sm=256.0,
        max_warps_per_sm=64,
    ),
    "h200": GPUProfile(
        name="h200",
        peak_hbm_gbs=4800.0,
        l2_mb=60.0,
        smem_kb_per_sm=228.0,
        registers_kb_per_sm=256.0,
        max_warps_per_sm=64,
    ),
}


def get_profile(device: str) -> GPUProfile:
    return GPU_PROFILES.get(device.lower(), GPU_PROFILES["cpu-validation"])


def softmax_bytes(batch: int, seq_len: int, dtype_bytes: int = 2, fused: bool = True) -> int:
    elements = batch * seq_len
    if fused:
        reads = elements * dtype_bytes
        writes = elements * dtype_bytes
        metadata = batch * 3 * 4
        return reads + writes + metadata
    reads = elements * dtype_bytes * 3
    writes = elements * dtype_bytes * 2
    return reads + writes


def occupancy_estimate(block_size: int, num_warps: int, profile: GPUProfile) -> float:
    smem_kb = max(block_size * 2 * 4 / 1024, 1.0)
    smem_limited = profile.smem_kb_per_sm / smem_kb if profile.smem_kb_per_sm else 1.0
    warp_limited = profile.max_warps_per_sm / max(num_warps, 1)
    resident_blocks = max(min(smem_limited, warp_limited, 8.0), 1.0)
    return round(min(resident_blocks * num_warps / profile.max_warps_per_sm, 1.0), 3)


def roofline_softmax(
    batch: int,
    seq_len: int,
    block_size: int,
    device: str = "rtx4090",
    num_warps: int = 4,
) -> dict[str, float | int | str]:
    profile = get_profile(device)
    fused_bytes = softmax_bytes(batch, seq_len, fused=True)
    eager_bytes = softmax_bytes(batch, seq_len, fused=False)
    occupancy = occupancy_estimate(block_size, num_warps, profile)
    block_efficiency = max(0.55, 1.0 - abs(block_size - 1024) / 4096)
    achieved_ratio = round(min(0.92, 0.35 + occupancy * 0.5 + block_efficiency * 0.15), 3)
    bandwidth_gbs = round(profile.peak_hbm_gbs * achieved_ratio, 2)
    fused_ms = round(fused_bytes / (bandwidth_gbs * 1e9) * 1000, 6)
    eager_ms = round(eager_bytes / (profile.peak_hbm_gbs * 0.35 * 1e9) * 1000, 6)
    return {
        "device": profile.name,
        "batch": batch,
        "seq_len": seq_len,
        "block_size": block_size,
        "num_warps": num_warps,
        "bytes_fused": fused_bytes,
        "bytes_eager": eager_bytes,
        "peak_hbm_gbs": profile.peak_hbm_gbs,
        "bandwidth_gbs": bandwidth_gbs,
        "occupancy": occupancy,
        "speedup_vs_torch": round(eager_ms / max(fused_ms, 1e-9), 3),
        "fused_estimated_ms": fused_ms,
        "torch_estimated_ms": eager_ms,
    }
