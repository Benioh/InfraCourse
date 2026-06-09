"""L38 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.rollout_manager")


def test_generate_uses_fresh_server_and_records_meta():
    impl = _impl()
    manager = impl.RolloutManager([impl.RolloutServer("s0", weight_version=3)], max_staleness=1)
    rollout = manager.generate(["hello"], actor_version=4)
    assert rollout.responses == ["response_from_s0_w3_0"]
    assert rollout.meta_info == {
        "server_id": "s0",
        "weight_version": 3,
        "actor_version": 4,
        "staleness": 1,
    }


def test_generate_skips_stale_server_round_robin():
    impl = _impl()
    manager = impl.RolloutManager(
        [impl.RolloutServer("stale", 0), impl.RolloutServer("fresh", 4)],
        max_staleness=1,
    )
    rollout = manager.generate(["p"], actor_version=5)
    assert rollout.meta_info["server_id"] == "fresh"
    assert manager.next_server_index == 0


def test_generate_raises_when_all_servers_stale():
    impl = _impl()
    manager = impl.RolloutManager([impl.RolloutServer("s0", 1)], max_staleness=1)
    with pytest.raises(RuntimeError):
        manager.generate(["p"], actor_version=5)


def test_update_weights_all_or_subset():
    impl = _impl()
    servers = [impl.RolloutServer("s0", 0), impl.RolloutServer("s1", 0)]
    manager = impl.RolloutManager(servers)
    assert manager.update_weights(3, server_ids=["s1"]) == ["s1"]
    assert [server.weight_version for server in servers] == [0, 3]
    assert manager.update_weights(4) == ["s0", "s1"]
    assert [server.weight_version for server in servers] == [4, 4]


def test_freshness_reports_actor_minus_engine_version():
    impl = _impl()
    manager = impl.RolloutManager([impl.RolloutServer("s0", 2), impl.RolloutServer("s1", 5)])
    assert manager.freshness(actor_version=6) == {"s0": 4, "s1": 1}
