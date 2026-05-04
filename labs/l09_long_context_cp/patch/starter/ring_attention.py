"""
L04.5 Patch · Ring Attention Forward (Online Softmax)

填空规则：
- TODO(student) 必须自己写
- 不许用 F.scaled_dot_product_attention 或 nn.MultiheadAttention
- 允许 torch.einsum / torch.exp / torch.maximum 等基础算子

完成度自检：
    make patch-test M=l09_long_context_cp
"""

from __future__ import annotations

import math

import torch


def ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: int = 1,
) -> torch.Tensor:
    """Online-softmax attention.

    q : (B, H, Sq, D)
    k : (B, H, Sk, D)
    v : (B, H, Sk, D)

    Returns: (B, H, Sq, D)
    """
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    # 切分 K, V 沿 seq 维。chunks 列表是 list[(k_chunk, v_chunk)]。
    # TODO(student): 用 torch.chunk(k, num_chunks, dim=2) / torch.chunk(v, num_chunks, dim=2)
    #   注意 torch.chunk 处理不整除时最后一块会更小，符合不变量 3。
    raise NotImplementedError("L04.5 Patch: implement chunking")

    # 初始化 running 状态：
    #   running_out  : (B, H, Sq, D)
    #   running_max  : (B, H, Sq, 1)
    #   running_denom: (B, H, Sq, 1)
    # TODO(student): 初始化这 3 个 tensor。max 用 -inf，其它用 0；dtype/device 跟 q 走。

    # for i, (k_chunk, v_chunk) in enumerate(...):
    #   1. scores = einsum('bhid,bhjd->bhij', q, k_chunk) * scale     # (B, H, Sq, chunk_len)
    #   2. chunk_max = scores.max(dim=-1, keepdim=True).values        # (B, H, Sq, 1)
    #   3. new_max = torch.maximum(running_max, chunk_max)
    #   4. exp_old = torch.exp(running_max - new_max)                 # 旧状态的 rescale 系数
    #   5. exp_chunk = torch.exp(scores - new_max)                    # 当前 chunk 的 exp
    #   6. running_denom = running_denom * exp_old + exp_chunk.sum(dim=-1, keepdim=True)
    #   7. running_out = running_out * exp_old + einsum('bhij,bhjd->bhid', exp_chunk, v_chunk)
    #   8. running_max = new_max

    # 最后 normalize：return running_out / running_denom
    raise NotImplementedError("L04.5 Patch: implement online softmax loop")
