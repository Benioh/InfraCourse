"""L01.5 Patch tests · 2-rank gloo, CPU OK."""

from __future__ import annotations

import os

from .conftest import parallel_run

WORLD_SIZE = int(os.environ.get("L01_5_WORLD_SIZE", "2"))


def test_grads_match_pytorch_ddp():
    with parallel_run(WORLD_SIZE, "grads_match_pytorch_ddp"):
        pass


def test_grads_are_averaged_not_summed():
    with parallel_run(WORLD_SIZE, "grads_are_averaged_not_summed"):
        pass


def test_world_size_1_is_noop():
    with parallel_run(1, "world_size_1_is_noop"):
        pass


def test_skips_no_grad_params():
    with parallel_run(WORLD_SIZE, "skips_no_grad_params"):
        pass


def test_handles_partial_grads():
    with parallel_run(WORLD_SIZE, "handles_partial_grads"):
        pass
