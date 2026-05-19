"""L30.5 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.cuda_graph_cache")


def test_first_call_captures():
    impl = _impl()
    gc = impl.GraphCache()
    x = torch.randn(8, 16)
    out = gc.capture_or_replay(lambda t: t * 2, x)
    assert gc.capture_count == 1
    assert gc.replay_count == 0
    assert torch.equal(out, x * 2)


def test_second_call_replays():
    impl = _impl()
    gc = impl.GraphCache()
    x = torch.randn(8, 16)
    gc.capture_or_replay(lambda t: t * 2, x)
    out2 = gc.capture_or_replay(lambda t: t * 2, x)
    assert gc.capture_count == 1
    assert gc.replay_count == 1
    assert torch.equal(out2, x * 2)


def test_different_shapes_distinct_graphs():
    impl = _impl()
    gc = impl.GraphCache()
    x = torch.randn(4, 4)
    y = torch.randn(8, 8)
    gc.capture_or_replay(lambda t: t + 1, x)
    gc.capture_or_replay(lambda t: t + 1, y)
    gc.capture_or_replay(lambda t: t + 1, x)  # replay
    assert gc.capture_count == 2
    assert gc.replay_count == 1


def test_replay_output_equal_to_eager():
    impl = _impl()
    gc = impl.GraphCache()
    fn = lambda t: torch.relu(t * 3 - 1)
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    out_capture = gc.capture_or_replay(fn, x)
    out_replay = gc.capture_or_replay(fn, x)
    out_eager = fn(x)
    assert torch.allclose(out_capture, out_eager)
    assert torch.allclose(out_replay, out_eager)


def test_savor_pause_records_metadata():
    impl = _impl()
    savor = impl.MemorySavor()
    t = torch.randn(1024)
    h = savor.pause(t)
    assert savor.is_paused(h)


def test_savor_resume_returns_same_data():
    impl = _impl()
    savor = impl.MemorySavor()
    torch.manual_seed(0)
    t = torch.randn(64, 8)
    h = savor.pause(t)
    restored = savor.resume(h)
    assert torch.equal(restored, t)


def test_total_paused_bytes_tracks_storage():
    impl = _impl()
    savor = impl.MemorySavor()
    t1 = torch.randn(1024, dtype=torch.float32)  # 4096 bytes
    t2 = torch.randn(512, dtype=torch.float32)   # 2048 bytes
    savor.pause(t1)
    savor.pause(t2)
    assert savor.total_paused_bytes() == 4096 + 2048


def test_resume_removes_from_paused_pool():
    impl = _impl()
    savor = impl.MemorySavor()
    t = torch.randn(100)
    h = savor.pause(t)
    assert savor.is_paused(h)
    savor.resume(h)
    assert not savor.is_paused(h)
    assert savor.total_paused_bytes() == 0
