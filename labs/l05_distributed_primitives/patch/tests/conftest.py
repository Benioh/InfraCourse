"""
Pytest harness for the L02 patch.

Key trick: we do NOT require torchrun / multi-process to test correctness.
Each test spawns N child processes via torch.multiprocessing.spawn,
inits a gloo process group, runs the assertions, and reports back via
a shared mp.Queue. This means the lab works on CPU only — the student
can run `make patch-test` locally without a GPU.

If TP_IMPL=reference is set, tests load the reference implementation
instead of the student starter (used by maintainers to validate the
test harness itself).
"""

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


def _get_impl_module():
    impl = os.environ.get("TP_IMPL", "starter")
    if impl == "reference":
        return importlib.import_module("reference.tp_linear")
    return importlib.import_module("starter.tp_linear")


def _worker(
    rank: int,
    world_size: int,
    port: int,
    fn_name: str,
    queue,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.manual_seed(0)
        # Late import — must be inside the worker so each process loads independently
        impl = _get_impl_module()
        from . import worker_cases  # type: ignore  # noqa: WPS433
        fn = getattr(worker_cases, fn_name)
        fn(rank, world_size, impl)
        if rank == 0:
            queue.put(("ok", None))
    except Exception as exc:  # pragma: no cover
        if rank == 0:
            queue.put(("fail", repr(exc)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def parallel_run(world_size: int, fn_name: str):
    """Spawn world_size workers, run fn_name in worker_cases, propagate failure."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker, args=(rank, world_size, port, fn_name, queue), daemon=False
        )
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                pytest.fail(f"worker {p.pid} timed out after 120s")
            if p.exitcode != 0:
                # rank 0 should have put failure on queue; pull it for context
                msg = "no message"
                if not queue.empty():
                    status, payload = queue.get_nowait()
                    msg = payload or status
                pytest.fail(f"worker exited with {p.exitcode}: {msg}")
        # Ensure rank 0 reported ok
        if not queue.empty():
            status, payload = queue.get_nowait()
            if status != "ok":
                pytest.fail(f"worker reported failure: {payload}")
        yield
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
