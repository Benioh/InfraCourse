"""
YaRN（Yet another RoPE extensioN）最小同构（L04.5 教学）。

教学目的：
    把 YaRN 的"频率分段缩放 + attention temperature 补偿"从 paper 公式
    变成可调用的函数，让学员在 n13 notebook 里：
        - 看 scale 随 target/train ratio 单调增长
        - 看 attention_temperature 随 scale 缓慢补偿（≈ 1 + 0.1*(scale-1)）
        - 验证训练时 ctx_train=4K，推理 32K，scale 应该是多少
    再回到 lab 跑真实 Megatron 的 YaRN 推理评估。

真实框架对照：
    - github_repo/Megatron-LM/megatron/core/models/common/embeddings/
      rotary_pos_embedding.py：YaRN 实现包含 NTK-aware 分段（低频不动、
      高频按 PI、中频插值）。本文件只用单一 log 缩放，简化但量级正确。
    - YaRN 论文：Peng et al., "YaRN: Efficient Context Window Extension
      of Large Language Models", 2023.

简化掉的复杂度：
    - 没有 NTK-aware 分段（low/mid/high 频段不同处理）
    - attention_factor 0.1 是教学启发式（真实约 0.07）
    - 没有 attention temperature 在 softmax 中的实际接入
"""
from __future__ import annotations

import math


def yarn_scale(train_seq_len: int, target_seq_len: int, beta_fast: float = 32.0) -> float:
    """YaRN scale 因子：target/train < 1 时不缩放，否则按 log 缓涨。

    简化版：scale = 1 + log(target/train) / log(beta_fast)
    真实 YaRN 把频率分段处理，这里用单一公式给出量级正确的预算值。
    """
    if train_seq_len <= 0 or target_seq_len <= 0:
        raise ValueError("sequence lengths must be positive")
    if target_seq_len <= train_seq_len:
        return 1.0
    ratio = target_seq_len / train_seq_len
    return round(1.0 + math.log(ratio) / math.log(beta_fast), 6)


def yarn_temperature(scale: float, attention_factor: float = 0.1) -> float:
    return round(1.0 + max(scale - 1.0, 0.0) * attention_factor, 6)


def yarn_summary(train_seq_len: int = 4096, target_seq_len: int = 32768) -> dict[str, float | int]:
    scale = yarn_scale(train_seq_len, target_seq_len)
    return {
        "train_seq_len": train_seq_len,
        "target_seq_len": target_seq_len,
        "yarn_scale": scale,
        "attention_temperature": yarn_temperature(scale),
        "validation_boundary": "math-only unless long-context eval perplexity is measured",
    }
