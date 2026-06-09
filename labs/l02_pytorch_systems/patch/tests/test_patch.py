"""L02 Patch tests · run with `make patch-test M=l02_pytorch_systems`. Pure CPU."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.memory_probe")


def make_mlp(in_dim=8, hidden=16, out_dim=4):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---- count_param_bytes ----


def test_count_param_bytes_matches_manual():
    impl = _impl()
    model = make_mlp()
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert impl.count_param_bytes(model) == expected


def test_works_with_mixed_dtype():
    """Half + float混合模型，字节数计算各 dtype 自适应。"""
    impl = _impl()
    model = nn.Sequential(
        nn.Linear(8, 16).half(),
        nn.Linear(16, 4),  # float
    )
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    actual = impl.count_param_bytes(model)
    assert actual == expected


# ---- count_grad_bytes ----


def test_count_grad_bytes_after_backward():
    impl = _impl()
    model = make_mlp()
    x = torch.randn(2, 8)
    model(x).sum().backward()
    # fp32 grad shape == param shape
    assert impl.count_grad_bytes(model) == impl.count_param_bytes(model)


def test_count_grad_bytes_zero_when_none():
    impl = _impl()
    model = make_mlp()
    # No backward — grads are None
    assert impl.count_grad_bytes(model) == 0
    # Run backward then zero out
    model(torch.randn(2, 8)).sum().backward()
    for p in model.parameters():
        p.grad = None
    assert impl.count_grad_bytes(model) == 0


# ---- count_optimizer_state_bytes ----


def test_optimizer_state_sgd_zero():
    impl = _impl()
    model = make_mlp()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    assert impl.count_optimizer_state_bytes(opt) == 0


def test_optimizer_state_sgd_momentum_one_x_params():
    impl = _impl()
    model = make_mlp()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    actual = impl.count_optimizer_state_bytes(opt)
    expected = impl.count_param_bytes(model)  # 1× (momentum_buffer)
    # Allow 5% slack for any per-param scalar state
    assert abs(actual - expected) < expected * 0.05


def test_optimizer_state_adam_two_x_params():
    impl = _impl()
    model = make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    actual = impl.count_optimizer_state_bytes(opt)
    expected = 2 * impl.count_param_bytes(model)  # 2× (exp_avg + exp_avg_sq)
    assert abs(actual - expected) < expected * 0.05
