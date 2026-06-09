"""Debug 方法论工具包 — starter 文件

实现 4 个函数，建立系统化的 debug 能力。
参考 patch/task.md 了解每个函数的详细规格。
"""

from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def tensor_diff_report(a, b) -> dict:
    """计算两个 tensor 的详细差异报告。

    返回 max_abs_diff, mean_abs_diff, max_rel_diff, mean_rel_diff, cosine_similarity。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def find_first_diverge_layer(baseline_outputs: list[dict], test_outputs: list[dict],
                             threshold: float = 0.999) -> dict:
    """逐层对比，找到第一个 cosine_similarity < threshold 的层。

    输入 list[dict]，每个 dict 包含 'name' 和 'tensor'。
    返回 dict 包含 diverge_layer_idx, diverge_layer_name, cosine_sim, all_aligned。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def make_minimal_repro_config(full_config: dict) -> dict:
    """从完整训练配置生成最小复现配置。

    缩小规则：
    - num_nodes → 1
    - num_gpus → 1 (如果原来 > 1 则设为 min(原值, 2))
    - batch_size → 1
    - max_seq_len → min(原值, 512)
    - num_steps → 5
    - seed → 42 (固定)
    - dataset_size → min(原值, 10)
    其他字段保留不变。
    """
    # TODO: 实现此函数
    raise NotImplementedError


def classify_symptom(symptom: dict) -> dict:
    """根据症状描述判断问题类别和建议排查方向。

    输入 symptom dict 包含：
    - type: str — 'loss_mismatch', 'hang', 'slow', 'oom', 'nan'
    - context: str — 额外上下文（如 'multi_gpu', 'packed', 'long_seq'）

    返回 dict 包含：
    - category: str — 问题大类
    - first_check: str — 第一步应该检查什么
    - likely_cause: str — 最可能的原因
    """
    # TODO: 实现此函数
    raise NotImplementedError
