"""Distributed harness for L12 bucketed grad sync."""

from __future__ import annotations

import importlib
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_impl():
    name = os.environ.get("IMPL") or "starter"
    return importlib.import_module(f"{name}.bucketed_ddp")


def _worker(rank, world_size, port, fn_name, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.manual_seed(0)
        impl = _get_impl()
        from . import worker_cases  # type: ignore  # noqa: WPS433
        getattr(worker_cases, fn_name)(rank, world_size, impl)
        if rank == 0:
            queue.put(("ok", None))
    except Exception as exc:
        if rank == 0:
            queue.put(("fail", repr(exc)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def parallel_run(world_size, fn_name):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, port, fn_name, queue), daemon=False)
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                pytest.fail(f"worker timed out")
            if p.exitcode != 0:
                msg = queue.get_nowait()[1] if not queue.empty() else "no msg"
                pytest.fail(f"worker exit {p.exitcode}: {msg}")
        yield
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
