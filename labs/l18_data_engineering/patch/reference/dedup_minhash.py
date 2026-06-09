"""Reference solution for L19 Patch · MinHash dedup."""

from __future__ import annotations

import hashlib
import random
from typing import List, Tuple


_PRIME = (1 << 61) - 1


def _shingles(text: str, n: int) -> set:
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _seeded_hashes(seed: int) -> Tuple[int, int]:
    rng = random.Random(seed)
    return (rng.randint(1, 2**31 - 1), rng.randint(0, 2**31 - 1))


def minhash_signature(text: str, num_perm: int = 64, n_gram: int = 3) -> List[int]:
    shingles = _shingles(text, n_gram)
    if not shingles:
        return [0] * num_perm
    base_hashes = []
    for s in shingles:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % _PRIME
        base_hashes.append(h)
    sig: List[int] = []
    for i in range(num_perm):
        a, b = _seeded_hashes(i)
        min_h = min((a * h + b) % _PRIME for h in base_hashes)
        sig.append(min_h)
    return sig


def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    assert len(sig1) == len(sig2)
    if not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def dedup(
    texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    n_gram: int = 3,
) -> List[int]:
    sigs = [minhash_signature(t, num_perm, n_gram) for t in texts]
    keep_indices: List[int] = []
    keep_sigs: List[List[int]] = []
    for i, sig in enumerate(sigs):
        is_dup = any(jaccard_estimate(sig, k) >= threshold for k in keep_sigs)
        if not is_dup:
            keep_indices.append(i)
            keep_sigs.append(sig)
    return keep_indices
