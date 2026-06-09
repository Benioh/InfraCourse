"""L00 Patch 测试：系统基础"""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "starter"))

from system_fundamentals import (
    estimate_h2d_transfer_time,
    identify_comm_bottleneck,
    dataloader_throughput,
    explain_pinned_memory,
)


class TestH2DTransfer:
    def test_pinned_faster(self):
        """pinned memory 传输比 pageable 快"""
        pinned = estimate_h2d_transfer_time(1_000_000_000, pinned=True)
        pageable = estimate_h2d_transfer_time(1_000_000_000, pinned=False)
        assert pinned["transfer_time_ms"] < pageable["transfer_time_ms"]
        assert pinned["bandwidth_gbps"] > pageable["bandwidth_gbps"]


class TestCommBottleneck:
    def test_nvlink_fastest(self):
        """同节点 GPU↔GPU 走 NVLink，比 PCIe 快"""
        nvlink = identify_comm_bottleneck("gpu_same_node", "gpu_same_node", 1_000_000_000)
        pcie = identify_comm_bottleneck("cpu", "gpu_same_node", 1_000_000_000)
        assert nvlink["bandwidth_gbps"] > pcie["bandwidth_gbps"]
        assert nvlink["path"] == "NVLink"

    def test_cross_node_ib(self):
        """跨节点走 InfiniBand"""
        ib = identify_comm_bottleneck("gpu_diff_node", "gpu_diff_node", 1_000_000_000)
        assert "InfiniBand" in ib["path"]


class TestDataloader:
    def test_throughput_scaling(self):
        """增加 workers 应该提高吞吐（直到 H2D 成为瓶颈）"""
        r1 = dataloader_throughput(1_000_000, num_workers=1, preprocessing_time_ms=100, h2d_time_ms=5)
        r4 = dataloader_throughput(1_000_000, num_workers=4, preprocessing_time_ms=100, h2d_time_ms=5)
        assert r4["throughput_batches_per_sec"] > r1["throughput_batches_per_sec"]
        assert r4["preprocessing_per_batch_ms"] < r1["preprocessing_per_batch_ms"]

    def test_h2d_bottleneck(self):
        """当 H2D 时间大于预处理时间，瓶颈是 H2D"""
        r = dataloader_throughput(1_000_000, num_workers=8, preprocessing_time_ms=10, h2d_time_ms=50)
        assert r["bottleneck"] == "h2d_transfer"


class TestPinnedMemory:
    def test_explain(self):
        """解释包含关键信息"""
        info = explain_pinned_memory()
        assert "DMA" in info["why_faster"] or "dma" in info["why_faster"].lower()
        assert "pin_memory" in info["pytorch_api"]
