"""Reference solution for L08.5 Patch · AWQ-lite quantization."""

from __future__ import annotations

import torch


def compute_awq_scale(
    weight: torch.Tensor,
    act_amax: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    eps = 1e-5
    w_amax_in = weight.abs().max(dim=0).values  # (in,)
    scale = act_amax.clamp(min=eps).pow(alpha) / w_amax_in.clamp(min=eps).pow(alpha)
    scale = scale.clamp(min=1.0)
    return scale


def quantize_w8_per_channel(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w_scaled = weight * scale.unsqueeze(0)
    per_out_amax = w_scaled.abs().max(dim=1).values
    per_out_scale = (per_out_amax / 127.0).clamp(min=1e-8)
    int_w = (
        (w_scaled / per_out_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    )
    return int_w, per_out_scale


def dequantize_w8_per_channel(
    int_weight: torch.Tensor,
    per_out_scale: torch.Tensor,
    awq_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    w = int_weight.float() * per_out_scale.unsqueeze(1)
    if awq_scale is not None:
        w = w / awq_scale.unsqueeze(0)
    return w
