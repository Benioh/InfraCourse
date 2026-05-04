"""
L02 Patch tests · run with `make patch-test M=l05_distributed_primitives`.

Pure-CPU, gloo-backed, world_size=2 by default. No GPU required.
Each test spawns 2 processes, joins them, and fails the test if any rank fails.
"""

from __future__ import annotations

import os

import pytest

from .conftest import parallel_run


WORLD_SIZE = int(os.environ.get("L02_WORLD_SIZE", "2"))


# ---- Numerical equivalence ----


def test_column_parallel_matches_single_gpu():
    with parallel_run(WORLD_SIZE, "column_parallel_matches_single_gpu"):
        pass


def test_row_parallel_matches_single_gpu():
    with parallel_run(WORLD_SIZE, "row_parallel_matches_single_gpu"):
        pass


# ---- Gradient correctness ----


def test_column_grad_matches_single_gpu():
    with parallel_run(WORLD_SIZE, "column_grad_matches_single_gpu"):
        pass


def test_row_grad_matches_single_gpu():
    with parallel_run(WORLD_SIZE, "row_grad_matches_single_gpu"):
        pass


# ---- Result-based behavior check ----
#
# We removed the call-counting tests (test_column_backward_triggers_allreduce,
# test_row_forward_triggers_allreduce) because they prescribe an implementation
# detail rather than a result. Any wrong communication pattern will already be
# caught by the gradient/forward equivalence tests above — if you skip the
# all-reduce, your numbers won't match nn.Linear, and grad/forward tests fail.
#
# We keep test_row_bias_added_once because it checks the OUTPUT (off by bias *
# world_size if implemented wrong), not the call sequence.


def test_row_bias_added_once():
    with parallel_run(WORLD_SIZE, "row_bias_added_once"):
        pass
