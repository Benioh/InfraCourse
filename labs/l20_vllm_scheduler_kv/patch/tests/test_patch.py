"""L21 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.scheduler")


def test_kv_cache_allocate_free_snapshot():
    impl = _impl()
    kv = impl.KVCacheManager(num_blocks=4)
    assert kv.allocate_slots("r0", 2) == [0, 1]
    snap = kv.snapshot()
    assert snap["usage"] == 0.5
    assert snap["used_blocks"] == {0: "r0", 1: "r0"}
    kv.free("r0")
    assert kv.snapshot()["free_blocks"] == [0, 1, 2, 3]


def test_kv_cache_exhaustion_raises():
    impl = _impl()
    kv = impl.KVCacheManager(num_blocks=1)
    kv.allocate_slots("r0", 1)
    with pytest.raises(RuntimeError):
        kv.allocate_slots("r1", 1)


def test_scheduler_moves_waiting_to_running_and_decode():
    impl = _impl()
    scheduler = impl.Scheduler(impl.KVCacheManager(num_blocks=4), max_num_running_reqs=2)
    scheduler.add_request(impl.Request("r0", "hello world", max_tokens=2))
    out = scheduler.schedule()
    assert out.scheduled_prefill == ["r0"]
    assert out.scheduled_decode == ["r0"]
    assert list(scheduler.running) == ["r0"]
    assert scheduler.kv_cache_manager.usage == 0.25


def test_scheduler_respects_max_running_and_leaves_waiting():
    impl = _impl()
    scheduler = impl.Scheduler(impl.KVCacheManager(num_blocks=8), max_num_running_reqs=1)
    scheduler.add_request(impl.Request("r0", "a", max_tokens=1))
    scheduler.add_request(impl.Request("r1", "b", max_tokens=1))
    out = scheduler.schedule()
    assert out.scheduled_prefill == ["r0"]
    assert [req.request_id for req in scheduler.waiting] == ["r1"]


def test_finished_request_frees_kv_blocks():
    impl = _impl()
    scheduler = impl.Scheduler(impl.KVCacheManager(num_blocks=4), max_num_running_reqs=1)
    request = impl.Request("r0", "a b c", max_tokens=1)
    scheduler.add_request(request)
    scheduler.schedule()
    request.output_tokens.append("done")
    out = scheduler.schedule()
    assert out.finished == ["r0"]
    assert "r0" in scheduler.finished
    assert scheduler.kv_cache_manager.usage == 0.0
