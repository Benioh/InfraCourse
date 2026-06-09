"""L40 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.gae_chunk")


def test_naive_known_values():
    impl = _impl()
    # 简单可手算的例子：γ=1, λ=1 → A_t = sum of (r - V_diff) from t to end
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 0.0])
    last_value = torch.tensor(0.0)
    out = impl.gae_naive(rewards, values, last_value, gamma=1.0, lam=1.0)
    # δ_2 = 1 + 0 - 0 = 1; A_2 = 1
    # δ_1 = 1 + 0 - 0 = 1; A_1 = 1 + 1*1 = 2
    # δ_0 = 1 + 0 - 0 = 1; A_0 = 1 + 1*2 = 3
    assert torch.allclose(out, torch.tensor([3.0, 2.0, 1.0]))


def test_chunked_matches_naive_short():
    impl = _impl()
    torch.manual_seed(0)
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T)
    last_value = torch.tensor(0.5)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.99, lam=0.95)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.99, lam=0.95, chunk_size=3)
    assert torch.allclose(a_naive, a_chunk, atol=1e-6), \
        f"max diff = {(a_naive - a_chunk).abs().max().item():.2e}"


def test_chunked_matches_naive_long():
    impl = _impl()
    torch.manual_seed(1)
    T = 1024
    rewards = torch.randn(T)
    values = torch.randn(T)
    last_value = torch.tensor(0.0)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.99, lam=0.95)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.99, lam=0.95, chunk_size=64)
    assert torch.allclose(a_naive, a_chunk, atol=1e-5), \
        f"长序列 max diff = {(a_naive - a_chunk).abs().max().item():.2e}"


def test_chunked_handles_remainder():
    impl = _impl()
    # T=11, chunk_size=4 → chunks of 4,4,3
    torch.manual_seed(2)
    T = 11
    rewards = torch.randn(T)
    values = torch.randn(T)
    last_value = torch.tensor(0.0)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.95, lam=0.9)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.95, lam=0.9, chunk_size=4)
    assert torch.allclose(a_naive, a_chunk, atol=1e-6)


def test_chunked_one_chunk_equals_naive():
    impl = _impl()
    torch.manual_seed(3)
    T = 5
    rewards = torch.randn(T)
    values = torch.randn(T)
    last_value = torch.tensor(0.0)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.99, lam=0.95)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.99, lam=0.95, chunk_size=100)
    assert torch.allclose(a_naive, a_chunk, atol=1e-6)


def test_terminal_value_propagates():
    impl = _impl()
    # 让 last_value 显著影响最后一个 chunk
    rewards = torch.zeros(8)
    values = torch.zeros(8)
    last_value = torch.tensor(10.0)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.99, lam=0.95)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.99, lam=0.95, chunk_size=3)
    assert torch.allclose(a_naive, a_chunk, atol=1e-6)
    # 也校验 last_value 的传播：A_7 = δ_7 = 0 + 0.99*10 - 0 = 9.9
    assert float(a_naive[7]) == pytest.approx(9.9, rel=1e-5)


def test_batched_input():
    impl = _impl()
    torch.manual_seed(4)
    B, T = 4, 32
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    last_value = torch.randn(B)
    a_naive = impl.gae_naive(rewards, values, last_value, gamma=0.99, lam=0.95)
    a_chunk = impl.gae_chunked_parallel(rewards, values, last_value, gamma=0.99, lam=0.95, chunk_size=8)
    assert a_chunk.shape == (B, T)
    assert torch.allclose(a_naive, a_chunk, atol=1e-6)
