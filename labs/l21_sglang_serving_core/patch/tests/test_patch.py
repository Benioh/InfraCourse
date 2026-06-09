"""L22 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.radix_cache")


def test_empty_cache_match_zero():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    assert cache.match_prefix([1, 2, 3]) == 0
    assert cache.total_tokens() == 0


def test_insert_then_match_full():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    cache.insert([10, 20, 30, 40])
    assert cache.match_prefix([10, 20, 30, 40]) == 4


def test_partial_prefix_match():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    cache.insert([1, 2, 3, 4, 5])
    assert cache.match_prefix([1, 2, 3]) == 3
    assert cache.match_prefix([1, 2, 9]) == 2  # 1, 2 match; 9 mismatch
    assert cache.match_prefix([99]) == 0


def test_repeated_insert_no_double_count():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    n1 = cache.insert([1, 2, 3])
    n2 = cache.insert([1, 2, 3])
    assert n1 == 3
    assert n2 == 0
    assert cache.total_tokens() == 3


def test_total_tokens_correct():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    cache.insert([1, 2, 3])
    cache.insert([1, 2, 4])  # shares [1, 2] prefix; 1 new node
    assert cache.total_tokens() == 4  # nodes: 1, 2, 3, 4


def test_evict_removes_lru():
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    cache.insert([1, 2, 3])
    cache.insert([10, 20, 30])
    assert cache.total_tokens() == 6
    # Touch [1, 2, 3] to make it most-recently-used
    cache.match_prefix([1, 2, 3])
    cache.evict(3)
    # [10, 20, 30] (older) should be gone; [1, 2, 3] preserved
    assert cache.match_prefix([1, 2, 3]) == 3
    assert cache.match_prefix([10, 20, 30]) == 0


def test_evict_preserves_shared_prefix():
    """If [1,2,3] and [1,2,4] both exist, evicting just 1 token kills only one leaf,
    keeping the shared [1, 2] alive."""
    impl = _impl()
    cache = impl.RadixCache(max_tokens=100)
    cache.insert([1, 2, 3])
    cache.insert([1, 2, 4])
    # touch [1, 2, 4] to mark [1, 2, 3] as LRU
    cache.match_prefix([1, 2, 4])
    cache.evict(1)
    # [1, 2, 3] losing 3, but [1, 2] still alive (still in [1, 2, 4])
    assert cache.match_prefix([1, 2]) == 2
    assert cache.match_prefix([1, 2, 4]) == 3
    # [1, 2, 3] no longer fully present
    assert cache.match_prefix([1, 2, 3]) == 2  # 1, 2 OK; 3 gone
