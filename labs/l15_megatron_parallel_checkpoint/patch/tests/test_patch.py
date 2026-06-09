"""L16 Patch tests · CPU only."""

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
    return importlib.import_module(f"{os.environ.get('IMPL') or 'starter'}.checkpointing")


def _states():
    return {
        "model_state": {"layer": "tiny"},
        "optimizer_state": {"distributed": True, "shards": [{"rank": 0}]},
        "scheduler_state": {"step_count": 12},
        "parallel_state": {"tp": 2, "pp": 1, "dp": 4},
    }


def test_save_checkpoint_writes_payload_and_latest_marker(tmp_path: Path):
    impl = _impl()
    states = _states()
    result = impl.save_checkpoint(tmp_path, iteration=12, **states)
    checkpoint_path = Path(result["checkpoint_path"])
    latest_marker = Path(result["latest_marker"])
    assert checkpoint_path.name == "iter_0000012.json"
    assert latest_marker.read_text(encoding="utf-8") == "12"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["format"] == impl.FORMAT
    assert payload["iteration"] == 12
    assert payload["scheduler_state"] == {"step_count": 12}


def test_load_checkpoint_roundtrip(tmp_path: Path):
    impl = _impl()
    states = _states()
    impl.save_checkpoint(tmp_path, 7, **states)
    payload = impl.load_checkpoint(tmp_path, expected_parallel_state={"tp": 2, "dp": 4})
    assert payload["iteration"] == 7
    assert payload["model_state"] == states["model_state"]
    assert payload["warnings"] == []


def test_strict_parallel_state_mismatch_raises(tmp_path: Path):
    impl = _impl()
    impl.save_checkpoint(tmp_path, 1, **_states())
    with pytest.raises(impl.CheckpointError):
        impl.load_checkpoint(tmp_path, expected_parallel_state={"tp": 4}, strict=True)


def test_non_strict_parallel_state_mismatch_returns_warning(tmp_path: Path):
    impl = _impl()
    impl.save_checkpoint(tmp_path, 1, **_states())
    payload = impl.load_checkpoint(tmp_path, expected_parallel_state={"tp": 4}, strict=False)
    assert len(payload["warnings"]) == 1
    assert "tp" in payload["warnings"][0]


def test_missing_latest_marker_raises(tmp_path: Path):
    impl = _impl()
    with pytest.raises(impl.CheckpointError):
        impl.load_checkpoint(tmp_path)
