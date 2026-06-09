"""L01.5 Patch 测试：GPU 执行模型探测"""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "starter"))

from gpu_exec_probe import (
    describe_dispatch_chain,
    measure_async_gap,
    classify_memory_hierarchy,
    classify_op_bottleneck,
)


class TestDispatchChain:
    def test_dispatch_chain_stages(self):
        """dispatch 链路包含 4 个阶段，顺序正确"""
        chain = describe_dispatch_chain("matmul")
        assert isinstance(chain, list)
        assert len(chain) == 4
        assert chain == ["python_call", "cpp_dispatch", "kernel_launch", "gpu_execute"]


class TestAsyncTiming:
    def test_async_timing_without_sync(self):
        """device=cpu 时 is_async 为 False"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        result = measure_async_gap(512, "cpu")
        assert isinstance(result, dict)
        assert "cpu_time_ms" in result
        assert "sync_time_ms" in result
        assert "is_async" in result
        assert result["is_async"] is False
        assert result["sync_time_ms"] == result["cpu_time_ms"]

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch")
        or not __import__("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_async_timing_with_sync(self):
        """device=cuda 时函数不 crash，且返回正确格式"""
        result = measure_async_gap(2048, "cuda")
        assert isinstance(result, dict)
        assert result["cpu_time_ms"] >= 0
        assert result["sync_time_ms"] >= result["cpu_time_ms"]


class TestMemoryHierarchy:
    def test_memory_hierarchy_bandwidth(self):
        """4 个层级，bandwidth 从高到低排列"""
        hierarchy = classify_memory_hierarchy()
        assert isinstance(hierarchy, list)
        assert len(hierarchy) == 4
        bandwidths = [h["relative_bandwidth"] for h in hierarchy]
        assert bandwidths == sorted(bandwidths, reverse=True)
        assert hierarchy[0]["name"] == "register"
        assert hierarchy[-1]["name"] == "hbm"
        assert hierarchy[-1]["relative_bandwidth"] == 1

    def test_memory_hierarchy_scope(self):
        """scope 值正确"""
        hierarchy = classify_memory_hierarchy()
        scope_map = {h["name"]: h["scope"] for h in hierarchy}
        assert scope_map["register"] == "per_thread"
        assert scope_map["shared_memory"] == "per_sm"
        assert scope_map["l2_cache"] == "global"
        assert scope_map["hbm"] == "global"


class TestBottleneck:
    def test_compute_vs_memory_bound(self):
        """matmul(4096,4096,4096) → compute_bound；elementwise(4096,4096,0) → memory_bound"""
        matmul_result = classify_op_bottleneck("matmul", 4096, 4096, 4096)
        assert matmul_result["bottleneck"] == "compute_bound"
        assert matmul_result["flops"] == 2 * 4096 * 4096 * 4096
        assert matmul_result["arithmetic_intensity"] > 100

        elem_result = classify_op_bottleneck("elementwise", 4096, 4096, 0)
        assert elem_result["bottleneck"] == "memory_bound"
        assert elem_result["flops"] == 4096 * 4096
        assert elem_result["arithmetic_intensity"] < 100

    def test_bottleneck_invalid_op(self):
        """非法 op_type 抛 ValueError"""
        with pytest.raises(ValueError, match="op_type"):
            classify_op_bottleneck("convolution", 128, 128, 3)
