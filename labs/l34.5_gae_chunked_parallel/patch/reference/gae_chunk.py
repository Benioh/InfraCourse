"""Reference solution for L40 · Chunked GAE."""

from __future__ import annotations

import torch


def gae_naive(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    T = rewards.shape[-1]
    advantages = torch.zeros_like(rewards)
    next_value = last_value
    next_adv = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        v_t = values[..., t]
        r_t = rewards[..., t]
        delta = r_t + gamma * next_value - v_t
        adv = delta + gamma * lam * next_adv
        advantages[..., t] = adv
        next_value = v_t
        next_adv = adv
    return advantages


def gae_chunked_parallel(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
    chunk_size: int,
) -> torch.Tensor:
    T = rewards.shape[-1]
    if chunk_size >= T:
        return gae_naive(rewards, values, last_value, gamma, lam)

    advantages = torch.zeros_like(rewards)
    num_chunks = (T + chunk_size - 1) // chunk_size

    next_value = last_value
    next_chunk_start_adv = torch.zeros_like(last_value)

    for c in reversed(range(num_chunks)):
        start = c * chunk_size
        end = min(start + chunk_size, T)
        next_adv_in_chunk = next_chunk_start_adv
        running_next_value = next_value
        for t in reversed(range(start, end)):
            v_t = values[..., t]
            r_t = rewards[..., t]
            delta = r_t + gamma * running_next_value - v_t
            adv = delta + gamma * lam * next_adv_in_chunk
            advantages[..., t] = adv
            running_next_value = v_t
            next_adv_in_chunk = adv
        next_chunk_start_adv = advantages[..., start]
        next_value = values[..., start]

    return advantages
