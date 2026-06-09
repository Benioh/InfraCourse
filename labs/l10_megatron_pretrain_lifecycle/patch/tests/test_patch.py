"""L11 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL') or 'starter'}.train_step")


class DummyOptimizer:
    def __init__(self, step_result=True):
        self.calls: list[str] = []
        self.step_result = step_result
        self.param_groups = [{"lr": 0.1}]

    def zero_grad(self):
        self.calls.append("zero_grad")

    def step(self):
        self.calls.append("step")
        return self.step_result


class DummyScheduler:
    def __init__(self):
        self.calls = 0
        self.lr = 0.1

    def step(self):
        self.calls += 1
        self.lr *= 0.5

    def get_lr(self):
        return self.lr


def test_train_step_call_order_and_metrics():
    impl = _impl()
    events = []

    def forward_backward(data_iterator, model):
        events.append(("forward_backward", next(data_iterator), model["name"]))
        return {"losses": [3.0, 1.0], "tokens": 8}

    optimizer = DummyOptimizer(step_result={"success": True, "grad_norm": 0.25})
    scheduler = DummyScheduler()
    metrics = impl.train_step(
        forward_backward, iter(["batch0"]), {"name": "tiny"}, optimizer, scheduler, 7
    )
    assert optimizer.calls == ["zero_grad", "step"]
    assert events == [("forward_backward", "batch0", "tiny")]
    assert scheduler.calls == 1
    assert metrics["iteration"] == 7
    assert metrics["loss"] == pytest.approx(2.0)
    assert metrics["num_microbatches"] == 2
    assert metrics["tokens"] == 8
    assert metrics["grad_norm"] == pytest.approx(0.25)
    assert metrics["skipped_iter"] == 0
    assert metrics["lr"] == pytest.approx(0.05)


def test_scheduler_not_stepped_when_optimizer_skips():
    impl = _impl()
    optimizer = DummyOptimizer(step_result=False)
    scheduler = DummyScheduler()
    metrics = impl.train_step(
        lambda _it, _model: {"loss": 5.0},
        iter(["batch0"]),
        {},
        optimizer,
        scheduler,
        1,
    )
    assert optimizer.calls == ["zero_grad", "step"]
    assert scheduler.calls == 0
    assert metrics["skipped_iter"] == 1
    assert metrics["lr"] == pytest.approx(0.1)


def test_accepts_optimizer_step_none_as_success():
    impl = _impl()
    optimizer = DummyOptimizer(step_result=None)
    scheduler = DummyScheduler()
    metrics = impl.train_step(lambda _it, _model: {"loss": 2.5}, iter([]), {}, optimizer, scheduler, 3)
    assert scheduler.calls == 1
    assert metrics["loss"] == pytest.approx(2.5)
    assert metrics["skipped_iter"] == 0


def test_rejects_missing_loss():
    impl = _impl()
    with pytest.raises(impl.TrainStepError):
        impl.train_step(lambda _it, _model: {"tokens": 8}, iter([]), {}, DummyOptimizer(), None, 1)


def test_rejects_empty_microbatch_losses():
    impl = _impl()
    with pytest.raises(impl.TrainStepError):
        impl.train_step(lambda _it, _model: {"losses": []}, iter([]), {}, DummyOptimizer(), None, 1)
