"""L08.7 Patch tests · CPU OK."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.spec_decode")


def _make_logits(argmax_seq, vocab=100):
    """Build target_logits where each row's argmax is argmax_seq[i]."""
    L = len(argmax_seq)
    out = torch.full((L, vocab), -10.0)
    for i, t in enumerate(argmax_seq):
        out[i, t] = 10.0  # large positive at desired argmax
    return out


def test_all_drafts_accepted():
    impl = _impl()
    drafts = [3, 7, 11, 23]
    target_logits = _make_logits([3, 7, 11, 23, 99])  # k+1 = 5 positions
    accepted, num = impl.greedy_verify(drafts, target_logits)
    assert num == 4
    assert accepted == [3, 7, 11, 23, 99]


def test_first_mismatch_at_position_2():
    impl = _impl()
    drafts = [3, 7, 11, 23]
    # First two match; position 2 target argmax is 99 (not 11) → reject from here
    target_logits = _make_logits([3, 7, 99, 0, 0])
    accepted, num = impl.greedy_verify(drafts, target_logits)
    assert num == 2
    assert accepted == [3, 7, 99]


def test_zero_drafts_one_bonus():
    impl = _impl()
    target_logits = _make_logits([42])
    accepted, num = impl.greedy_verify([], target_logits)
    assert num == 0
    assert accepted == [42]


def test_all_mismatch():
    impl = _impl()
    drafts = [1, 2, 3]
    target_logits = _make_logits([99, 0, 0, 0])  # position 0 mismatches immediately
    accepted, num = impl.greedy_verify(drafts, target_logits)
    assert num == 0
    assert accepted == [99]


def test_output_dtypes():
    impl = _impl()
    drafts = [1, 2]
    target_logits = _make_logits([1, 2, 3])
    accepted, num = impl.greedy_verify(drafts, target_logits)
    assert isinstance(accepted, list)
    assert all(isinstance(t, int) for t in accepted)
    assert isinstance(num, int)
