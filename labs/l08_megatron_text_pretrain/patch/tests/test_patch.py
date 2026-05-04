"""L04 Patch tests · pure CPU."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.lr_scheduler")


def _make_optim():
    p = [torch.nn.Parameter(torch.zeros(3))]
    return torch.optim.SGD(p, lr=0.0)


def test_initial_lr_is_max():
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, max_lr=1e-3, min_lr=1e-5,
                                       restart_steps=[1000, 3000], total_steps=5000)
    assert sched.get_lr() == pytest.approx(1e-3)


def test_lr_decreases_within_segment():
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [1000, 3000], 5000)
    prev = sched.get_lr()
    for _ in range(500):
        sched.step()
        cur = sched.get_lr()
        assert cur <= prev + 1e-12, f"lr should decrease in segment; {prev} → {cur}"
        prev = cur


def test_lr_at_restart_step_is_max():
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [1000, 3000], 5000)
    for _ in range(1000):
        sched.step()
    # at step 1000 (the restart_step), lr should reset to max_lr
    assert sched.get_lr() == pytest.approx(1e-3, abs=1e-9)


def test_lr_clamps_after_total():
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [1000, 3000], 5000)
    for _ in range(6000):
        sched.step()
    assert sched.get_lr() == pytest.approx(1e-5)


def test_multiple_param_groups_synced():
    impl = _impl()
    p1 = torch.nn.Parameter(torch.zeros(3))
    p2 = torch.nn.Parameter(torch.zeros(5))
    opt = torch.optim.SGD([{"params": [p1]}, {"params": [p2]}], lr=0.0)
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [1000], 2000)
    for _ in range(500):
        sched.step()
    lrs = [g["lr"] for g in opt.param_groups]
    assert all(abs(lrs[0] - lr) < 1e-12 for lr in lrs), f"all groups should sync; got {lrs}"


def test_get_lr_matches_optimizer():
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [], 1000)
    for _ in range(500):
        sched.step()
    assert sched.get_lr() == opt.param_groups[0]["lr"]


def test_no_restarts_is_pure_cosine():
    """No restarts → behaves like standard cosine over [0, total_steps]."""
    impl = _impl()
    opt = _make_optim()
    sched = impl.CosineWithRestartsLR(opt, 1e-3, 1e-5, [], 1000)
    # at step 500 (midpoint), cosine value = 0.5 * (1 + cos(π * 0.5)) = 0.5
    for _ in range(500):
        sched.step()
    expected = 1e-5 + (1e-3 - 1e-5) * 0.5  # cos(π/2) = 0 → factor = 0.5
    assert sched.get_lr() == pytest.approx(expected, rel=1e-3)
