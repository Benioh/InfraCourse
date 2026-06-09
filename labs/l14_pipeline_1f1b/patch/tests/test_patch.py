"""L15 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL') or 'starter'}.pp_schedule")


def test_warmup_lengths():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for stage, timeline in enumerate(schedule):
        warmup = 4 - stage - 1
        prefix = timeline[:warmup]
        assert all(op == "F" for op, _ in prefix), (stage, prefix)


def test_steady_alternates_F_B():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for stage, timeline in enumerate(schedule):
        warmup = 4 - stage - 1
        steady_len = 2 * (8 - warmup)
        steady = timeline[warmup : warmup + steady_len]
        for idx, (op, _) in enumerate(steady):
            expected = "F" if idx % 2 == 0 else "B"
            assert op == expected, (stage, idx, op)


def test_cooldown_lengths():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for stage, timeline in enumerate(schedule):
        warmup = 4 - stage - 1
        cooldown = timeline[-warmup:] if warmup > 0 else []
        assert all(op == "B" for op, _ in cooldown)


def test_each_microbatch_has_one_F_one_B_per_stage():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for timeline in schedule:
        forwards = [m for op, m in timeline if op == "F"]
        backwards = [m for op, m in timeline if op == "B"]
        assert sorted(forwards) == list(range(8))
        assert sorted(backwards) == list(range(8))


def test_microbatch_order_per_stage():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for timeline in schedule:
        forwards = [m for op, m in timeline if op == "F"]
        backwards = [m for op, m in timeline if op == "B"]
        assert forwards == sorted(forwards)
        assert backwards == sorted(backwards)


def test_no_microbatch_backward_before_its_forward():
    impl = _impl()
    schedule = impl.make_1f1b_schedule(num_stages=4, num_microbatches=8)
    for timeline in schedule:
        f_pos = {m: idx for idx, (op, m) in enumerate(timeline) if op == "F"}
        for idx, (op, m) in enumerate(timeline):
            if op == "B":
                assert idx > f_pos[m]


def test_bubble_count_2_times_pp_minus_1():
    impl = _impl()
    assert impl.bubble_count(1) == 0
    assert impl.bubble_count(4) == 6
    assert impl.bubble_count(8) == 14


def test_invalid_num_microbatches_raises():
    impl = _impl()
    with pytest.raises(ValueError):
        impl.make_1f1b_schedule(num_stages=4, num_microbatches=2)
