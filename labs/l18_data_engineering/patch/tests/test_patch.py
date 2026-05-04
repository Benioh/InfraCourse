"""L06.3 Patch tests · CPU OK."""

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
    return importlib.import_module(f"{name}.dedup_minhash")


def test_signature_size():
    impl = _impl()
    sig = impl.minhash_signature("hello world this is a test", num_perm=128)
    assert len(sig) == 128


def test_identical_text_same_signature():
    impl = _impl()
    text = "the quick brown fox jumps over the lazy dog"
    sig1 = impl.minhash_signature(text, num_perm=64)
    sig2 = impl.minhash_signature(text, num_perm=64)
    assert sig1 == sig2
    assert impl.jaccard_estimate(sig1, sig2) == 1.0


def test_jaccard_estimate_high_for_near_duplicates():
    impl = _impl()
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox jumps over the lazy cat"  # 1 word changed
    sig_a = impl.minhash_signature(a, num_perm=128)
    sig_b = impl.minhash_signature(b, num_perm=128)
    j = impl.jaccard_estimate(sig_a, sig_b)
    assert j > 0.7, f"near-duplicate jaccard = {j} (expected > 0.7)"

    # Completely different texts → low jaccard
    c = "in computer science a hash function is any function"
    sig_c = impl.minhash_signature(c, num_perm=128)
    j_ac = impl.jaccard_estimate(sig_a, sig_c)
    assert j_ac < 0.2, f"distinct text jaccard = {j_ac} (expected < 0.2)"


def test_dedup_removes_exact_duplicates():
    impl = _impl()
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",  # dup
        "in computer science a hash function is any function",
    ]
    keep = impl.dedup(texts, threshold=0.8, num_perm=64)
    assert keep == [0, 2], f"expected [0, 2]; got {keep}"


def test_dedup_preserves_distinct():
    impl = _impl()
    texts = [
        "alpha beta gamma delta epsilon",
        "the quick brown fox jumps",
        "in computer science a hash",
        "lorem ipsum dolor sit amet consectetur",
    ]
    keep = impl.dedup(texts, threshold=0.8, num_perm=64)
    assert keep == [0, 1, 2, 3], f"distinct texts should all be kept; got {keep}"
