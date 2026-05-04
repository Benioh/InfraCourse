"""
L08.5 Patch · AWQ-lite per-channel quantization

填空规则：
- TODO(student) 必须自己写
- 不许 import bitsandbytes / auto-gptq
- 允许 torch 基础算子

完成度自检：
    make patch-test M=l23_quant_serving
"""

from __future__ import annotations

import torch


def compute_awq_scale(
    weight: torch.Tensor,    # (out_features, in_features)
    act_amax: torch.Tensor,  # (in_features,) per-input-channel activation amax
    alpha: float = 0.5,
) -> torch.Tensor:
    """返回 per-input-channel scale（用于把 outlier act "吸" 到 weight 里）.

    注意：scale 是 in_features 维度的（不是 out_features！），因为 activation 沿 in_features
    流入。返回 shape (in_features,)。
    """
    # TODO(student):
    #   w_amax_in = weight.abs().max(dim=0).values  # 每输入通道的 weight amax (in_features,)
    #   eps = 1e-5
    #   scale = (act_amax.clamp(min=eps).pow(alpha) / w_amax_in.clamp(min=eps).pow(alpha))
    #   scale = scale.clamp(min=1.0)  # 不允许缩小（保持数值稳定）
    #   return scale
    raise NotImplementedError("L08.5: implement compute_awq_scale")


def quantize_w8_per_channel(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 weight * scale 做 per-output-channel 对称 INT8 量化。

    Args:
        weight: (out, in)
        scale:  (in,)  来自 compute_awq_scale

    Returns:
        int_weight:  (out, in) int8
        per_out_scale: (out,) fp32 — 每输出通道的 dequantize 系数
    """
    # TODO(student):
    #   w_scaled = weight * scale.unsqueeze(0)  # (out, in)
    #   per_out_amax = w_scaled.abs().max(dim=1).values  # (out,)
    #   per_out_scale = per_out_amax / 127.0
    #   per_out_scale = per_out_scale.clamp(min=1e-8)
    #   int_w = (w_scaled / per_out_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    #   return int_w, per_out_scale
    raise NotImplementedError("L08.5: implement quantize_w8_per_channel")


def dequantize_w8_per_channel(
    int_weight: torch.Tensor,    # (out, in) int8
    per_out_scale: torch.Tensor, # (out,)
    awq_scale: torch.Tensor | None = None,  # (in,) 如果有，回退 awq scaling
) -> torch.Tensor:
    """还原 fp32 weight。需要同时除回 awq_scale 才能得到原始 weight."""
    # TODO(student):
    #   w = int_weight.float() * per_out_scale.unsqueeze(1)   # 还原 W * awq_scale
    #   if awq_scale is not None:
    #       w = w / awq_scale.unsqueeze(0)                   # 还原 W
    #   return w
    raise NotImplementedError("L08.5: implement dequantize_w8_per_channel")
