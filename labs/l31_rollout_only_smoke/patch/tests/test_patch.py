"""L35 Patch tests · CPU OK."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.rollout_pool")


def test_basic_rollout():
    impl = _impl()

    async def gen(p):
        return f"echo:{p}"

    pool = impl.RolloutPool(gen, max_concurrency=2)
    out = asyncio.run(pool.rollout(["a", "b", "c"]))
    assert out == ["echo:a", "echo:b", "echo:c"]


def test_preserves_order():
    """Different latencies; output must still match input order."""
    impl = _impl()

    async def gen(p):
        # short prompts get longer delays — done order != input order
        delays = {"a": 0.05, "b": 0.01, "c": 0.03}
        await asyncio.sleep(delays.get(p, 0.0))
        return p.upper()

    pool = impl.RolloutPool(gen, max_concurrency=10)
    out = asyncio.run(pool.rollout(["a", "b", "c"]))
    assert out == ["A", "B", "C"]


def test_concurrency_bounded():
    """Track max concurrent invocations; must not exceed max_concurrency."""
    impl = _impl()
    counter = {"running": 0, "max_running": 0}
    lock = asyncio.Lock()

    async def gen(p):
        async with lock:
            counter["running"] += 1
            counter["max_running"] = max(counter["max_running"], counter["running"])
        await asyncio.sleep(0.01)
        async with lock:
            counter["running"] -= 1
        return p

    pool = impl.RolloutPool(gen, max_concurrency=3)
    asyncio.run(pool.rollout([f"p{i}" for i in range(20)]))
    assert counter["max_running"] <= 3, f"max_running = {counter['max_running']}, expected ≤ 3"


def test_empty_input():
    impl = _impl()

    async def gen(p):
        return p

    pool = impl.RolloutPool(gen, max_concurrency=4)
    out = asyncio.run(pool.rollout([]))
    assert out == []


def test_propagates_errors():
    impl = _impl()

    async def gen(p):
        if p == "bad":
            raise ValueError("boom")
        return p

    pool = impl.RolloutPool(gen, max_concurrency=4)
    with pytest.raises(ValueError, match="boom"):
        asyncio.run(pool.rollout(["a", "bad", "c"]))
