"""L36 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.weight_sync")


def _make_pair(train_data, inference_data):
    train_state = dict(train_data)
    inference_state = dict(inference_data)

    def setter(new):
        inference_state.update(new)

    return train_state, inference_state, setter


def test_basic_sync():
    impl = _impl()
    torch.manual_seed(0)
    train, inf, setter = _make_pair(
        {"a": torch.randn(4, 8), "b": torch.randn(2, 2)},
        {"a": torch.zeros(4, 8), "b": torch.zeros(2, 2)},
    )
    coord = impl.WeightSyncCoordinator(
        train_state_provider=lambda: train,
        inference_state_setter=setter,
        inference_state_provider=lambda: inf,
    )
    stats = coord.sync()
    assert torch.equal(inf["a"], train["a"])
    assert torch.equal(inf["b"], train["b"])
    assert stats["mismatched_keys"] == []
    assert stats["num_tensors"] == 2


def test_shape_mismatch_skipped():
    impl = _impl()
    train, inf, setter = _make_pair(
        {"a": torch.randn(4, 8), "b": torch.randn(2, 2)},
        {"a": torch.zeros(4, 8), "b": torch.zeros(3, 3)},  # b shape mismatch
    )
    inf_b_before = inf["b"].clone()
    coord = impl.WeightSyncCoordinator(
        lambda: train, setter, inference_state_provider=lambda: inf
    )
    stats = coord.sync()
    assert "b" in stats["mismatched_keys"]
    assert torch.equal(inf["a"], train["a"])
    # b should not have changed
    assert torch.equal(inf["b"], inf_b_before)
    assert stats["num_tensors"] == 1


def test_dtype_mismatch_skipped():
    impl = _impl()
    train, inf, setter = _make_pair(
        {"a": torch.randn(4, 8, dtype=torch.float32)},
        {"a": torch.zeros(4, 8, dtype=torch.float16)},
    )
    coord = impl.WeightSyncCoordinator(
        lambda: train, setter, inference_state_provider=lambda: inf
    )
    stats = coord.sync()
    assert "a" in stats["mismatched_keys"]
    assert stats["num_tensors"] == 0


def test_extra_train_keys_skipped():
    impl = _impl()
    train, inf, setter = _make_pair(
        {"a": torch.randn(4, 8), "extra": torch.randn(3, 3)},
        {"a": torch.zeros(4, 8)},
    )
    coord = impl.WeightSyncCoordinator(
        lambda: train, setter, inference_state_provider=lambda: inf
    )
    stats = coord.sync()
    assert "extra" in stats["mismatched_keys"]
    assert "extra" not in inf
    assert torch.equal(inf["a"], train["a"])


def test_stats_correct():
    impl = _impl()
    train, inf, setter = _make_pair(
        {"a": torch.randn(4, 8, dtype=torch.float32), "b": torch.randn(2, 2)},
        {"a": torch.zeros(4, 8), "b": torch.zeros(2, 2)},
    )
    coord = impl.WeightSyncCoordinator(
        lambda: train, setter, inference_state_provider=lambda: inf
    )
    stats = coord.sync()
    expected_bytes = 4 * 8 * 4 + 2 * 2 * 4  # both fp32
    assert stats["bytes_synced"] == expected_bytes
    assert stats["num_tensors"] == 2
