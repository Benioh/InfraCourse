"""L40 Patch · GAE chunked parallel.

填空规则：
- TODO(student) 必须自己写
- 禁止 import 任何已实现的 GAE
- 允许 torch.zeros_like / 切片 / for-loop

完成度自检：
    make patch-test M=l34.5_gae_chunked_parallel
"""

from __future__ import annotations

import torch


def gae_naive(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """反向递推 baseline。

    支持 (T,) 与 (B, T) 两种 shape；time 维度永远是最后一维。

    A_t = δ_t + γλ A_{t+1},   δ_t = r_t + γ V_{t+1} - V_t,   A_T = 0
    """
    # TODO(student):
    #   T = rewards.shape[-1]
    #   advantages = torch.zeros_like(rewards)
    #   next_value = last_value
    #   next_adv = torch.zeros_like(last_value)
    #   for t in reversed(range(T)):
    #       v_t = values[..., t]
    #       r_t = rewards[..., t]
    #       delta = r_t + gamma * next_value - v_t
    #       adv = delta + gamma * lam * next_adv
    #       advantages[..., t] = adv
    #       next_value = v_t
    #       next_adv = adv
    #   return advantages
    raise NotImplementedError("L40: implement gae_naive")


def gae_chunked_parallel(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
    chunk_size: int,
) -> torch.Tensor:
    """分 chunk 计算 GAE，与 naive 在数值上 allclose(atol=1e-6).

    步骤：
      1. T = rewards.shape[-1]; 把序列切成 chunk_size 长度的 chunks（最后一个可能更短）。
      2. 反向遍历 chunk c (从最后一个 chunk 往前)：
           - 在 chunk 内部反向递推：以 A_{end_of_chunk} = next_chunk_start_adv 为初值
           - 记下 A_at_chunk_start，作为 c-1 轮的 next_chunk_start_adv
      3. 拼接所有 chunk 的 A_local → 全局 advantages。

    并行化点：chunk 内部递推**互相独立**——可以在 GPU 上把所有 chunk 同时算
    （这版我们写顺序版，但保证数学等价；学生可以挑战写真正并行的 batched 版）。
    """
    # TODO(student):
    #   T = rewards.shape[-1]
    #   advantages = torch.zeros_like(rewards)
    #
    #   if chunk_size >= T:
    #       return gae_naive(rewards, values, last_value, gamma, lam)
    #
    #   num_chunks = (T + chunk_size - 1) // chunk_size
    #   next_value = last_value
    #   next_chunk_start_adv = torch.zeros_like(last_value)
    #
    #   for c in reversed(range(num_chunks)):
    #       start = c * chunk_size
    #       end = min(start + chunk_size, T)
    #       # chunk 内部反向递推
    #       next_adv_in_chunk = next_chunk_start_adv
    #       running_next_value = next_value
    #       for t in reversed(range(start, end)):
    #           v_t = values[..., t]
    #           r_t = rewards[..., t]
    #           delta = r_t + gamma * running_next_value - v_t
    #           adv = delta + gamma * lam * next_adv_in_chunk
    #           advantages[..., t] = adv
    #           running_next_value = v_t
    #           next_adv_in_chunk = adv
    #       # 这个 chunk 起点的 adv，作为前一个 chunk 的 boundary
    #       next_chunk_start_adv = advantages[..., start]
    #       # 同样，这个 chunk 起点的 value，作为前一个 chunk 的 next_value
    #       next_value = values[..., start]
    #
    #   return advantages
    raise NotImplementedError("L40: implement gae_chunked_parallel")
