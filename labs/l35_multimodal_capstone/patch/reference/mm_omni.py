"""Reference solution for L41 Capstone Patch · MM-Tiny-Omni components."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    def __init__(
        self,
        vit_dim: int,
        whisper_dim: int,
        llm_dim: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Linear(vit_dim, llm_dim)
        self.audio_proj = nn.Linear(whisper_dim, llm_dim)
        self.image_pos_emb = nn.Parameter(torch.zeros(num_image_tokens, llm_dim))
        nn.init.normal_(self.image_pos_emb, std=0.02)

    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> dict:
        out = {"image_emb": None, "audio_emb": None}
        if image_features is not None:
            img = self.image_proj(image_features)
            img = img + self.image_pos_emb.unsqueeze(0)
            out["image_emb"] = img
        if audio_features is not None:
            out["audio_emb"] = self.audio_proj(audio_features)
        return out


def wer_score(hypothesis: str, reference: str) -> float:
    hyp = hypothesis.split()
    ref = reference.split()
    n, m = len(hyp), len(ref)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / max(1, m)


def clip_score(image_embed: torch.Tensor, text_embed: torch.Tensor) -> float:
    norm_a = image_embed / (image_embed.norm() + 1e-9)
    norm_b = text_embed / (text_embed.norm() + 1e-9)
    return float((norm_a * norm_b).sum().item())
