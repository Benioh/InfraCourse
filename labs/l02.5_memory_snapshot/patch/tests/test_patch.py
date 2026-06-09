"""L03 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL") or "starter"
    return importlib.import_module(f"{name}.memory_snapshot")


def test_disabled_tracker_returns_invalid_addr():
    impl = _impl()
    tr = impl.MemoryTracker()
    addr = tr.alloc(1024, ("test_stack",))
    assert addr == -1
    snap = tr.dump_snapshot()
    assert snap["total_leaked_bytes"] == 0


def test_basic_alloc_free_balance():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    addr = tr.alloc(1024, ("frame_a",))
    assert addr > 0
    tr.free(addr)
    snap = tr.dump_snapshot()
    assert snap["total_leaked_bytes"] == 0
    assert len(snap["live_allocations"]) == 0


def test_alloc_without_free_leaks():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    addr = tr.alloc(2048, ("frame_b",))
    assert addr > 0
    snap = tr.dump_snapshot()
    assert snap["total_leaked_bytes"] == 2048
    assert len(snap["live_allocations"]) == 1


def test_double_free_is_safe():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    addr = tr.alloc(1024, ("x",))
    tr.free(addr)
    tr.free(addr)  # must not raise
    tr.free(99999999)  # never-allocated addr, also safe
    snap = tr.dump_snapshot()
    assert snap["total_leaked_bytes"] == 0


def test_dump_snapshot_shape():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    tr.alloc(100, ("a",))
    tr.alloc(200, ("b",))
    snap = tr.dump_snapshot()
    assert "events" in snap
    assert "live_allocations" in snap
    assert "total_leaked_bytes" in snap
    assert snap["total_leaked_bytes"] == 300
    assert len(snap["live_allocations"]) == 2
    # events should record both allocs (no frees yet)
    assert sum(1 for kind, _ in snap["events"] if kind == "alloc") == 2


def test_find_top_leaks_groups_by_stack():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    stack_a = ("call_a:1", "main:1")
    stack_b = ("call_b:1", "main:2")
    tr.alloc(1024, stack_a)
    tr.alloc(1024, stack_a)  # 2 × 1024 from same stack → 2048
    tr.alloc(8192, stack_b)  # 8192 from different stack
    snap = tr.dump_snapshot()
    top = impl.find_top_leaks_by_stack(snap, k=2)
    assert len(top) == 2
    # biggest first
    assert top[0][1] == 8192
    assert top[0][0] == stack_b
    assert top[1][1] == 2048
    assert top[1][0] == stack_a


def test_find_top_leaks_respects_k():
    impl = _impl()
    tr = impl.MemoryTracker()
    tr.start()
    for i in range(5):
        tr.alloc(100 * (i + 1), (f"stack_{i}",))
    snap = tr.dump_snapshot()
    top2 = impl.find_top_leaks_by_stack(snap, k=2)
    assert len(top2) == 2
    # Top should be 500, 400 (descending)
    assert top2[0][1] == 500
    assert top2[1][1] == 400


def test_get_caller_stack_returns_tuple_of_strings():
    impl = _impl()
    stack = impl.get_caller_stack(depth=2)
    assert isinstance(stack, tuple)
    assert all(isinstance(s, str) for s in stack)
    # current test function should appear in the first frame
    assert "test_get_caller_stack_returns_tuple_of_strings" in stack[0]
