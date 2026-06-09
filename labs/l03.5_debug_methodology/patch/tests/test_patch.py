"""L03.5 Patch 测试：Debug 方法论工具包"""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "starter"))

from debug_toolkit import (
    tensor_diff_report,
    find_first_diverge_layer,
    make_minimal_repro_config,
    classify_symptom,
)


class TestTensorDiff:
    def test_tensor_diff_report(self):
        """能计算两个 tensor 的完整差异报告"""
        import torch
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        report = tensor_diff_report(a, b)
        assert report["max_abs_diff"] == 0.0
        assert report["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)

        c = torch.tensor([1.0, 2.0, 3.0, 5.0])
        report2 = tensor_diff_report(a, c)
        assert report2["max_abs_diff"] == 1.0
        assert report2["cosine_similarity"] < 1.0
        assert report2["cosine_similarity"] > 0.9


class TestLayerAlignment:
    def test_layer_alignment(self):
        """能逐层对比，找到第一个 diverge 层"""
        import torch
        baseline = [
            {"name": "embedding", "tensor": torch.randn(10, 64)},
            {"name": "layer_0_attn", "tensor": torch.randn(10, 64)},
            {"name": "layer_0_mlp", "tensor": torch.randn(10, 64)},
        ]
        test_good = [
            {"name": "embedding", "tensor": baseline[0]["tensor"].clone()},
            {"name": "layer_0_attn", "tensor": baseline[1]["tensor"].clone()},
            {"name": "layer_0_mlp", "tensor": baseline[2]["tensor"].clone()},
        ]
        result = find_first_diverge_layer(baseline, test_good)
        assert result["all_aligned"] is True
        assert result["diverge_layer_idx"] == -1

        test_bad = [
            {"name": "embedding", "tensor": baseline[0]["tensor"].clone()},
            {"name": "layer_0_attn", "tensor": torch.randn(10, 64)},
            {"name": "layer_0_mlp", "tensor": torch.randn(10, 64)},
        ]
        result2 = find_first_diverge_layer(baseline, test_bad, threshold=0.99)
        assert result2["all_aligned"] is False
        assert result2["diverge_layer_idx"] == 1
        assert result2["diverge_layer_name"] == "layer_0_attn"


class TestMinimalRepro:
    def test_minimal_repro_config(self):
        """能从完整配置中生成最小复现配置"""
        full = {
            "num_nodes": 4,
            "num_gpus": 8,
            "batch_size": 32,
            "max_seq_len": 8192,
            "num_steps": 10000,
            "seed": 123,
            "dataset_size": 100000,
            "model_name": "llama-7b",
            "lr": 1e-4,
        }
        minimal = make_minimal_repro_config(full)
        assert minimal["num_nodes"] == 1
        assert minimal["num_gpus"] == 2
        assert minimal["batch_size"] == 1
        assert minimal["max_seq_len"] == 512
        assert minimal["num_steps"] == 5
        assert minimal["seed"] == 42
        assert minimal["dataset_size"] == 10
        assert minimal["model_name"] == "llama-7b"
        assert minimal["lr"] == 1e-4


class TestClassifySymptom:
    def test_classify_symptom(self):
        """能根据症状判断问题类别"""
        result = classify_symptom({"type": "hang", "context": "multi_gpu"})
        assert result["category"] == "distributed_sync"
        assert "NCCL" in result["first_check"]

        result2 = classify_symptom({"type": "loss_mismatch", "context": "packed"})
        assert result2["category"] == "data_alignment"
        assert "attention_mask" in result2["first_check"]

        result3 = classify_symptom({"type": "oom", "context": ""})
        assert result3["category"] == "memory"

        result4 = classify_symptom({"type": "nan", "context": ""})
        assert result4["category"] == "numerical_stability"

    def test_diverge_detection(self):
        """slow + multi_gpu 应该建议查 overlap"""
        result = classify_symptom({"type": "slow", "context": "multi_gpu"})
        assert result["category"] == "performance"
        assert "overlap" in result["likely_cause"] or "communication" in result["likely_cause"]
