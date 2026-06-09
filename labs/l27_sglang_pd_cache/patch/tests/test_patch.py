"""L28 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(
        f"{os.environ.get('IMPL', 'starter')}.disagg_service"
    )


def test_route_request_records_worker_choice_and_kv_transfer():
    impl = _impl()
    service = impl.DisaggregationService(
        prefill_workers=["prefill-0"],
        decode_workers=["decode-0"],
    )
    route = service.route_request(
        "req-1", prompt_token_count=128, cached_prefix_tokens=32
    )
    assert route == impl.DisaggRoute("req-1", "prefill-0", "decode-0", 128, 32, 96, 128)
    assert service.transfers == [impl.KVTransfer("req-1", "prefill-0", "decode-0", 128)]
    assert service.metrics()["worker_loads"] == {"prefill-0": 96, "decode-0": 128}


def test_route_request_selects_least_loaded_workers():
    impl = _impl()
    service = impl.DisaggregationService(
        prefill_workers=["prefill-0", "prefill-1"],
        decode_workers=["decode-0", "decode-1"],
    )
    service.route_request("req-1", prompt_token_count=100)
    service.route_request("req-2", prompt_token_count=40)
    second = service.routes["req-2"]
    assert second.prefill_worker == "prefill-1"
    assert second.decode_worker == "decode-1"


def test_route_request_validates_required_fields():
    service = _impl().DisaggregationService()
    with pytest.raises(ValueError):
        service.route_request("", 1)
    with pytest.raises(ValueError):
        service.route_request("req", 0)
    with pytest.raises(ValueError):
        service.route_request("req", 8, cached_prefix_tokens=9)


def test_metrics_aggregate_cache_prefill_and_decode_worker_tokens():
    service = _impl().DisaggregationService(
        prefill_workers=["prefill-0", "prefill-1"],
        decode_workers=["decode-0", "decode-1"],
    )
    service.route_request("req-1", prompt_token_count=128, cached_prefix_tokens=64)
    service.route_request("req-2", prompt_token_count=64, cached_prefix_tokens=0)
    service.route_request("req-3", prompt_token_count=32, cached_prefix_tokens=16)
    metrics = service.metrics()
    assert metrics == {
        "kv_transfers": 3,
        "tokens_transferred": 224,
        "prefill_tokens": 144,
        "cached_prefix_tokens": 80,
        "active_requests": 3,
        "tokens_by_prefill_worker": {"prefill-0": 160, "prefill-1": 64},
        "tokens_by_decode_worker": {"decode-0": 128, "decode-1": 96},
        "worker_loads": {
            "prefill-0": 80,
            "prefill-1": 64,
            "decode-0": 128,
            "decode-1": 96,
        },
    }


def test_transfers_for_request_filters_in_order():
    service = _impl().DisaggregationService()
    first = service.transfer_kv("req-1", "p0", "d0", 10)
    service.transfer_kv("req-2", "p0", "d0", 20)
    second = service.transfer_kv("req-1", "p1", "d1", 30)
    assert service.transfers_for_request("req-1") == [first, second]


def test_complete_request_releases_active_worker_load():
    service = _impl().DisaggregationService()
    service.route_request("req-1", prompt_token_count=10, cached_prefix_tokens=3)
    assert service.metrics()["active_requests"] == 1
    service.complete_request("req-1")
    assert service.metrics()["active_requests"] == 0
    assert service.metrics()["worker_loads"] == {"prefill-0": 0, "decode-0": 0}
    service.complete_request("unknown")


def test_empty_metrics_are_zero():
    assert _impl().DisaggregationService().metrics() == {
        "kv_transfers": 0,
        "tokens_transferred": 0,
        "prefill_tokens": 0,
        "cached_prefix_tokens": 0,
        "active_requests": 0,
        "tokens_by_prefill_worker": {},
        "tokens_by_decode_worker": {},
        "worker_loads": {"prefill-0": 0, "decode-0": 0},
    }
