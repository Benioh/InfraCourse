"""L27 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.metrics_exporter")


def test_gauge_set_and_export():
    impl = _impl()
    m = impl.MetricsExporter()
    m.set_gauge("prefill_queue_depth", 12)
    out = m.export()
    assert "# TYPE prefill_queue_depth gauge" in out
    assert "prefill_queue_depth 12" in out


def test_counter_increment():
    impl = _impl()
    m = impl.MetricsExporter()
    m.inc_counter("requests_total")
    m.inc_counter("requests_total")
    m.inc_counter("requests_total", by=3)
    out = m.export()
    assert "# TYPE requests_total counter" in out
    # value = 5
    assert "requests_total 5" in out


def test_hit_rate_calculation():
    impl = _impl()
    m = impl.MetricsExporter()
    m.record_event("prefix_cache", hit=True)
    m.record_event("prefix_cache", hit=True)
    m.record_event("prefix_cache", hit=True)
    m.record_event("prefix_cache", hit=False)
    out = m.export()
    assert "# TYPE prefix_cache_hit_rate gauge" in out
    # 3 / (3+1) = 0.75
    assert "prefix_cache_hit_rate 0.75" in out


def test_zero_events_returns_zero():
    impl = _impl()
    m = impl.MetricsExporter()
    m.record_event("foo", hit=False)  # only miss, 0 hits
    # but we want zero division safety: try empty as well
    out = m.export()
    # 0 / 1 = 0.0
    assert "foo_hit_rate 0.0" in out


def test_labels_sorted():
    impl = _impl()
    m = impl.MetricsExporter()
    m.set_gauge("test", 1.0, labels={"z": "1", "a": "2", "m": "3"})
    out = m.export()
    # labels should be alphabetically ordered: a, m, z
    assert 'test{a="2",m="3",z="1"} 1.0' in out
