"""Debug 方法论工具包 — 参考实现"""

from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def tensor_diff_report(a, b) -> dict:
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (b.abs() + 1e-8)
    cosine_sim = F.cosine_similarity(
        a.flatten().unsqueeze(0).float(),
        b.flatten().unsqueeze(0).float(),
    ).item()

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_similarity": cosine_sim,
    }


def find_first_diverge_layer(baseline_outputs: list[dict], test_outputs: list[dict],
                             threshold: float = 0.999) -> dict:
    num_layers = min(len(baseline_outputs), len(test_outputs))

    for i in range(num_layers):
        a = baseline_outputs[i]["tensor"].flatten().unsqueeze(0).float()
        b = test_outputs[i]["tensor"].flatten().unsqueeze(0).float()
        cos = F.cosine_similarity(a, b).item()

        if cos < threshold:
            return {
                "diverge_layer_idx": i,
                "diverge_layer_name": baseline_outputs[i]["name"],
                "cosine_sim": cos,
                "all_aligned": False,
            }

    return {
        "diverge_layer_idx": -1,
        "diverge_layer_name": None,
        "cosine_sim": 1.0,
        "all_aligned": True,
    }


def make_minimal_repro_config(full_config: dict) -> dict:
    result = dict(full_config)
    result["num_nodes"] = 1
    result["num_gpus"] = min(full_config.get("num_gpus", 1), 2)
    if result["num_gpus"] <= 1:
        result["num_gpus"] = 1
    result["batch_size"] = 1
    result["max_seq_len"] = min(full_config.get("max_seq_len", 512), 512)
    result["num_steps"] = 5
    result["seed"] = 42
    result["dataset_size"] = min(full_config.get("dataset_size", 10), 10)
    return result


def classify_symptom(symptom: dict) -> dict:
    stype = symptom["type"]
    context = symptom.get("context", "")

    if stype == "loss_mismatch":
        if "packed" in context:
            return {
                "category": "data_alignment",
                "first_check": "attention_mask and position_ids after packing",
                "likely_cause": "packed samples not properly isolated in attention mask",
            }
        return {
            "category": "data_alignment",
            "first_check": "loss mask and label alignment",
            "likely_cause": "loss reduction or mask mismatch between baseline and test",
        }
    elif stype == "hang":
        return {
            "category": "distributed_sync",
            "first_check": "NCCL_DEBUG=INFO logs for which rank is stuck",
            "likely_cause": "not all ranks reaching the same collective operation",
        }
    elif stype == "slow":
        if "multi_gpu" in context:
            return {
                "category": "performance",
                "first_check": "nsys timeline for communication overlap",
                "likely_cause": "communication not overlapping with computation",
            }
        return {
            "category": "performance",
            "first_check": "profiler for GPU utilization and top ops",
            "likely_cause": "GPU idle due to CPU overhead or data loading",
        }
    elif stype == "oom":
        return {
            "category": "memory",
            "first_check": "torch.cuda.memory_summary() and memory profiler",
            "likely_cause": "activation memory or retained computation graph",
        }
    elif stype == "nan":
        return {
            "category": "numerical_stability",
            "first_check": "gradient norm and loss scale history",
            "likely_cause": "loss scale too high or exploding gradients in mixed precision",
        }
    else:
        return {
            "category": "unknown",
            "first_check": "minimal reproduction with fixed seed",
            "likely_cause": "insufficient information to classify",
        }
