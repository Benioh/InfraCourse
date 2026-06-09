"""L17 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL') or 'starter'}.crash_safe")


def test_atomic_save_writes_tmp_then_rename(tmp_path: Path):
    impl = _impl()
    target = tmp_path / "x.pt"
    impl.atomic_save({"a": 1}, target)
    assert target.exists()
    # tmp file must be cleaned up after rename
    assert not (tmp_path / "x.pt.tmp").exists()
    assert json.loads(target.read_text())["a"] == 1


def test_partial_tmp_only_is_discarded(tmp_path: Path):
    impl = _impl()
    impl.save_step(tmp_path, step=1, model_state={"w": 1}, optimizer_state={}, rng_state={})
    impl.save_step(tmp_path, step=2, model_state={"w": 2}, optimizer_state={}, rng_state={})
    # simulate a crashed save_step at step=3 — only .tmp present
    bogus_tmp = tmp_path / "iter_0000003.pt.tmp"
    bogus_tmp.write_text("PARTIAL")
    payload = impl.load_latest(tmp_path)
    assert not bogus_tmp.exists(), "tmp should be cleaned up"
    assert payload["step"] == 2


def test_resume_restores_step_rng_optimizer(tmp_path: Path):
    impl = _impl()
    impl.save_step(
        tmp_path,
        step=42,
        model_state={"w": [1.0, 2.0]},
        optimizer_state={"momentum": [0.1, 0.2]},
        rng_state={"seed": 12345},
        extra={"loss_history": [3.0, 2.5, 2.1]},
    )
    payload = impl.load_latest(tmp_path)
    assert payload["step"] == 42
    assert payload["optimizer_state"]["momentum"] == [0.1, 0.2]
    assert payload["rng_state"]["seed"] == 12345
    assert payload["extra"]["loss_history"][-1] == 2.1


def test_save_skips_when_step_unchanged(tmp_path: Path):
    impl = _impl()
    impl.save_step(tmp_path, step=5, model_state={"w": 1}, optimizer_state={}, rng_state={})
    files_before = sorted(tmp_path.glob("iter_*.pt"))
    impl.save_step(tmp_path, step=5, model_state={"w": 2}, optimizer_state={}, rng_state={})
    files_after = sorted(tmp_path.glob("iter_*.pt"))
    assert len(files_before) == len(files_after) == 1


def test_load_when_no_checkpoint_returns_none(tmp_path: Path):
    impl = _impl()
    assert impl.load_latest(tmp_path) is None


def test_bit_exact_loss_continuation(tmp_path: Path):
    impl = _impl()

    def step(state, grad):
        state["w"] = state["w"] - 0.1 * grad
        state["m"] = 0.9 * state["m"] + 0.1 * grad
        return state["w"] ** 2

    # baseline
    state_b = {"w": 5.0, "m": 0.0}
    grads = [0.5, -0.2, 0.3, 0.1, -0.4, 0.6, -0.1, 0.05, 0.2, -0.3]
    losses_b: list[float] = []
    for g in grads:
        losses_b.append(step(state_b, g))

    # crash run: stop after 4 steps, save, then resume
    state_c = {"w": 5.0, "m": 0.0}
    losses_c: list[float] = []
    for g in grads[:4]:
        losses_c.append(step(state_c, g))
    impl.save_step(
        tmp_path,
        step=4,
        model_state={"w": state_c["w"]},
        optimizer_state={"m": state_c["m"]},
        rng_state={},
    )
    payload = impl.load_latest(tmp_path)
    state_r = {
        "w": payload["model_state"]["w"],
        "m": payload["optimizer_state"]["m"],
    }
    assert state_r == state_c
    for g in grads[4:]:
        losses_c.append(step(state_r, g))
    assert losses_b == pytest.approx(losses_c, rel=1e-9)
