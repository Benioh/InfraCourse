"""L12.5 Patch 测试：并行策略分析"""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "starter"))

from parallel_analysis import (
    estimate_training_memory,
    compute_comm_volume,
    select_parallel_strategy,
    zero_memory_breakdown,
)


class TestMemoryEstimate:
    def test_memory_estimate(self):
        """7B FP16 Adam 模型的显存估算合理"""
        result = estimate_training_memory(7.0, precision="fp16", optimizer="adam")
        assert result["params_gb"] == pytest.approx(7 * 2 / 1.073741824, rel=0.01)
        assert result["total_gb"] > result["params_gb"]
        assert result["optimizer_gb"] > result["params_gb"]


class TestCommVolume:
    def test_comm_volume_allreduce(self):
        """all-reduce 通信量 = 2N(ws-1)/ws"""
        r = compute_comm_volume("all_reduce", 1000000, 8)
        assert r["comm_bytes"] == 2 * 1000000 * 7 // 8
        assert r["formula"] == "2 * N * (ws-1) / ws"

    def test_comm_volume_reduce_scatter(self):
        """reduce-scatter 通信量 = N(ws-1)/ws"""
        r = compute_comm_volume("reduce_scatter", 1000000, 8)
        assert r["comm_bytes"] == 1000000 * 7 // 8
        assert r["formula"] == "N * (ws-1) / ws"


class TestStrategySelection:
    def test_strategy_selection_small(self):
        """1B 模型 + 80GB GPU → DDP"""
        r = select_parallel_strategy(1.0, num_gpus=8, gpu_memory_gb=80.0)
        assert r["strategy"] == "DDP"

    def test_strategy_selection_large(self):
        """70B 模型 + 80GB GPU + 单机 → TP + FSDP"""
        r = select_parallel_strategy(70.0, num_gpus=8, gpu_memory_gb=80.0, inter_node=False)
        assert "TP" in r["strategy"]


class TestZeroBreakdown:
    def test_zero_memory_saving(self):
        """ZeRO-3 比 ZeRO-1 省更多显存"""
        z1 = zero_memory_breakdown(7.0, world_size=8, zero_stage=1)
        z2 = zero_memory_breakdown(7.0, world_size=8, zero_stage=2)
        z3 = zero_memory_breakdown(7.0, world_size=8, zero_stage=3)

        assert z1["saving_vs_ddp_pct"] > 0
        assert z2["saving_vs_ddp_pct"] > z1["saving_vs_ddp_pct"]
        assert z3["saving_vs_ddp_pct"] > z2["saving_vs_ddp_pct"]
        assert z3["total_per_gpu_gb"] < z1["total_per_gpu_gb"]
