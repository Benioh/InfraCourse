"""L02.7 Patch 测试：Profiler 数据解析与瓶颈分类"""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "starter"))

from profiler_analyzer import (
    parse_op_summary,
    classify_bottleneck,
    detect_gpu_idle_segments,
    memory_breakdown,
)


SAMPLE_EVENTS = [
    {"name": "aten::mm", "cpu_time_us": 200, "cuda_time_us": 5000, "calls": 10, "input_shapes": [[4096, 4096]]},
    {"name": "aten::add", "cpu_time_us": 50, "cuda_time_us": 800, "calls": 100, "input_shapes": [[4096]]},
    {"name": "aten::layer_norm", "cpu_time_us": 80, "cuda_time_us": 1200, "calls": 24, "input_shapes": [[1, 2048, 4096]]},
    {"name": "aten::softmax", "cpu_time_us": 30, "cuda_time_us": 600, "calls": 24, "input_shapes": [[1, 32, 2048, 2048]]},
    {"name": "aten::copy_", "cpu_time_us": 10, "cuda_time_us": 200, "calls": 50, "input_shapes": [[4096]]},
    {"name": "aten::mul", "cpu_time_us": 40, "cuda_time_us": 400, "calls": 80, "input_shapes": [[4096]]},
]


class TestOpSummary:
    def test_parse_op_summary(self):
        """top-K 正确、排序正确、pct 之和合理"""
        result = parse_op_summary(SAMPLE_EVENTS, top_k=3)
        assert len(result) == 3
        assert result[0]["name"] == "aten::mm"
        assert result[1]["name"] == "aten::layer_norm"
        assert result[2]["name"] == "aten::add"

        assert result[0]["cuda_time_us"] == 5000
        assert result[0]["calls"] == 10
        assert result[0]["avg_cuda_time_us"] == 500.0

        total_pct = sum(r["pct"] for r in result)
        assert total_pct <= 100.0
        assert result[0]["pct"] > result[1]["pct"]


class TestBottleneck:
    def test_classify_bottleneck_compute(self):
        """matmul + high util → compute_bound"""
        stats = {
            "gpu_utilization": 0.92,
            "top_op_type": "matmul",
            "avg_kernel_gap_us": 5.0,
            "comm_pct": 5.0,
            "data_wait_pct": 2.0,
        }
        result = classify_bottleneck(stats)
        assert result["bottleneck"] == "compute_bound"
        assert result["confidence"] == "high"

    def test_classify_bottleneck_memory(self):
        """elementwise + high util → memory_bound"""
        stats = {
            "gpu_utilization": 0.85,
            "top_op_type": "elementwise",
            "avg_kernel_gap_us": 8.0,
            "comm_pct": 10.0,
            "data_wait_pct": 3.0,
        }
        result = classify_bottleneck(stats)
        assert result["bottleneck"] == "memory_bound"
        assert result["confidence"] == "high"

    def test_classify_bottleneck_launch(self):
        """large gap + low util → launch_bound"""
        stats = {
            "gpu_utilization": 0.45,
            "top_op_type": "elementwise",
            "avg_kernel_gap_us": 120.0,
            "comm_pct": 5.0,
            "data_wait_pct": 3.0,
        }
        result = classify_bottleneck(stats)
        assert result["bottleneck"] == "launch_bound"
        assert result["confidence"] == "high"


class TestGpuIdle:
    def test_detect_gpu_idle(self):
        """正确识别超过阈值的 idle 段"""
        timeline = [
            {"start_us": 0, "end_us": 100, "name": "kernel_a"},
            {"start_us": 110, "end_us": 200, "name": "kernel_b"},
            {"start_us": 500, "end_us": 700, "name": "kernel_c"},
            {"start_us": 710, "end_us": 900, "name": "kernel_d"},
        ]
        segments = detect_gpu_idle_segments(timeline, threshold_us=100.0)
        assert len(segments) == 1
        assert segments[0]["before_kernel"] == "kernel_b"
        assert segments[0]["after_kernel"] == "kernel_c"
        assert segments[0]["duration_us"] == 300.0
        assert segments[0]["start_us"] == 200
        assert segments[0]["end_us"] == 500

    def test_check_comm_overlap(self):
        """timeline 中通信和计算的间隙小于阈值说明 overlap"""
        timeline = [
            {"start_us": 0, "end_us": 100, "name": "compute_fwd"},
            {"start_us": 105, "end_us": 200, "name": "nccl_allreduce"},
            {"start_us": 202, "end_us": 350, "name": "compute_bwd"},
        ]
        segments = detect_gpu_idle_segments(timeline, threshold_us=50.0)
        assert len(segments) == 0


class TestMemoryBreakdown:
    def test_memory_breakdown(self):
        """正确分类、百分比之和约等于 100"""
        allocations = [
            {"size_bytes": 1000000, "category": "parameter", "is_live": True},
            {"size_bytes": 1000000, "category": "gradient", "is_live": True},
            {"size_bytes": 2000000, "category": "optimizer_state", "is_live": True},
            {"size_bytes": 3000000, "category": "activation", "is_live": True},
            {"size_bytes": 500000, "category": "activation", "is_live": False},
            {"size_bytes": 500000, "category": "other", "is_live": True},
        ]
        result = memory_breakdown(allocations)
        assert result["total_live_bytes"] == 7500000
        assert result["largest_category"] == "activation"
        assert result["breakdown"]["activation"] == 3000000
        assert result["breakdown"]["optimizer_state"] == 2000000
        assert abs(sum(result["breakdown_pct"].values()) - 100.0) < 0.01
