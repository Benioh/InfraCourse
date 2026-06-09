"""
L41 Capstone Patch · MM-Tiny-Omni Components

填空规则：
- TODO(student) 必须自己写
- 不许用 jiwer / clip 等现成库做 wer / cosine
- 允许 torch.* / 基础算子

完成度自检：
    make patch-test M=l35_multimodal_capstone
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalProjector(nn.Module):
    """把 ViT / Whisper 输出投影到 LLM hidden dim。

    forward 同时支持 image-only / audio-only / 两者并存。
    """

    def __init__(
        self,
        vit_dim: int,
        whisper_dim: int,
        llm_dim: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        # TODO(student):
        #   self.image_proj = nn.Linear(vit_dim, llm_dim)
        #   self.audio_proj = nn.Linear(whisper_dim, llm_dim)
        #   self.image_pos_emb = nn.Parameter(torch.zeros(num_image_tokens, llm_dim))
        #   nn.init.normal_(self.image_pos_emb, std=0.02)
        raise NotImplementedError("L41: implement __init__")

    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """image_features: (B, num_image_tokens, vit_dim) 或 None
           audio_features: (B, T, whisper_dim) 或 None
           返回 {"image_emb": (B, N, llm_dim), "audio_emb": (B, T, llm_dim)} —— 缺的模态返回 None。
        """
        # TODO(student):
        #   out = {"image_emb": None, "audio_emb": None}
        #   if image_features is not None:
        #       img = self.image_proj(image_features)              # (B, N, llm_dim)
        #       img = img + self.image_pos_emb.unsqueeze(0)        # 加位置 embed
        #       out["image_emb"] = img
        #   if audio_features is not None:
        #       out["audio_emb"] = self.audio_proj(audio_features)
        #   return out
        raise NotImplementedError("L41: implement forward")


def wer_score(hypothesis: str, reference: str) -> float:
    """Word Error Rate based on Levenshtein 编辑距离 / 参考词数。

    规则：
      - hypothesis 和 reference 各按空格 split 成 word list
      - 算 word-level edit distance（插入 / 删除 / 替换 各 1 cost）
      - 返回 distance / max(1, len(reference_words))
      - 完全相同 → 0.0
    """
    hyp = hypothesis.split()
    ref = reference.split()
    # TODO(student):
    #   实现经典 Levenshtein DP：
    #     dp[i][j] = 把 hyp[:i] 转成 ref[:j] 的最少操作数
    #     dp[0][j] = j（全是 insert）
    #     dp[i][0] = i（全是 delete）
    #     dp[i][j] = dp[i-1][j-1] if hyp[i-1] == ref[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    #   distance = dp[len(hyp)][len(ref)]
    #   return distance / max(1, len(ref))
    raise NotImplementedError("L41: implement wer_score")


def clip_score(image_embed: torch.Tensor, text_embed: torch.Tensor) -> float:
    """归一化余弦相似度。两个 embed shape (D,)。返回 [-1, 1]。

    cos(a, b) = (a · b) / (||a|| × ||b||)
    """
    # TODO(student):
    #   norm_a = image_embed / (image_embed.norm() + 1e-9)
    #   norm_b = text_embed / (text_embed.norm() + 1e-9)
    #   return (norm_a * norm_b).sum().item()
    raise NotImplementedError("L41: implement clip_score")
