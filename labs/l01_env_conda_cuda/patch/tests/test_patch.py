from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("IMPL", "starter")
sys.path.insert(0, str(LAB_ROOT / IMPL))

import env_probe  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    yield


REQUIRED_KEYS = {
    "python_version",
    "python_executable",
    "pythonpath",
    "cuda_visible_devices",
    "local_rank",
    "world_size",
}


def test_collect_has_required_keys():
    payload = env_probe.collect_python_env()
    missing = REQUIRED_KEYS - set(payload.keys())
    assert not missing, f"missing required keys: {missing}"


def test_collect_pythonpath_is_list(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    payload = env_probe.collect_python_env()
    assert isinstance(payload["pythonpath"], list)
    assert payload["pythonpath"] == []

    monkeypatch.setenv("PYTHONPATH", f"/a{os.pathsep}/b{os.pathsep}")
    payload = env_probe.collect_python_env()
    assert payload["pythonpath"] == ["/a", "/b"]


def test_collect_cvd_unset_vs_empty(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert env_probe.collect_python_env()["cuda_visible_devices"] is None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert env_probe.collect_python_env()["cuda_visible_devices"] == ""


def test_parse_no_cvd_returns_local_rank():
    env = {"cuda_visible_devices": None}
    out = env_probe.parse_local_rank_mapping(env, local_rank=2)
    assert out["physical_device"] == 2
    assert out["visible_devices"] == []
    assert out["local_rank"] == 2


def test_parse_remapped_cards():
    env = {"cuda_visible_devices": "2,3"}
    out = env_probe.parse_local_rank_mapping(env, local_rank=1)
    assert out["physical_device"] == 3
    assert out["visible_devices"] == [2, 3]


def test_parse_out_of_bounds_raises():
    env = {"cuda_visible_devices": "0,1"}
    with pytest.raises(IndexError) as excinfo:
        env_probe.parse_local_rank_mapping(env, local_rank=5)
    msg = str(excinfo.value)
    assert "local_rank=" in msg
    assert "visible_devices=" in msg
    assert "out of range" in msg


def test_drift_pythonpath_change():
    base = {
        "pythonpath": ["/usr/lib/python3"],
        "cuda_visible_devices": None,
    }
    cur = {
        "pythonpath": ["/usr/lib/python3", "/home/me/repo"],
        "cuda_visible_devices": None,
    }
    diff = env_probe.summarize_drift(base, cur)
    assert len(diff) == 1
    assert diff[0].startswith("pythonpath:")
    assert "/home/me/repo" in diff[0]
