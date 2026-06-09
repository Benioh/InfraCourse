"""
L19 Patch · MinHash 文本去重

填空规则：
- TODO(student) 必须自己写
- 不许 import datasketch / sklearn
- 允许 hashlib / set / random 等 stdlib

完成度自检：
    make patch-test M=l18_data_engineering
"""

from __future__ import annotations

import hashlib
import random
from typing import List


def _shingles(text: str, n: int) -> set:
    """char-level n-gram shingles."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _seeded_hashes(seed: int) -> tuple[int, int]:
    """从 seed 生成两个大常数 (a, b) 用于 universal hash a*x + b mod p."""
    rng = random.Random(seed)
    return (rng.randint(1, 2**31 - 1), rng.randint(0, 2**31 - 1))


_PRIME = (1 << 61) - 1  # Mersenne prime


def minhash_signature(text: str, num_perm: int = 64, n_gram: int = 3) -> List[int]:
    """计算 text 的 MinHash 签名（长度 = num_perm）。

    实现思路：
        1. shingles = _shingles(text, n_gram)
        2. 对每个 shingle 用 hashlib.md5 算一个稳定基础哈希 h(s)
        3. 对 num_perm 个不同的 (a_i, b_i) seed：
             sig[i] = min((a_i * h(s) + b_i) mod p for s in shingles)
        4. return sig
    """
    # TODO(student):
    #   shingles = _shingles(text, n_gram)
    #   if not shingles: return [0] * num_perm
    #
    #   # 把每个 shingle 哈希成一个 64-bit int
    #   base_hashes = []
    #   for s in shingles:
    #       h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % _PRIME
    #       base_hashes.append(h)
    #
    #   sig = []
    #   for i in range(num_perm):
    #       a, b = _seeded_hashes(i)
    #       min_h = min((a * h + b) % _PRIME for h in base_hashes)
    #       sig.append(min_h)
    #   return sig
    raise NotImplementedError("L19: implement minhash_signature")


def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """估算两个 MinHash 签名的 Jaccard 相似度。"""
    # TODO(student):
    #   assert len(sig1) == len(sig2)
    #   if not sig1: return 0.0
    #   matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    #   return matches / len(sig1)
    raise NotImplementedError("L19: implement jaccard_estimate")


def dedup(
    texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    n_gram: int = 3,
) -> List[int]:
    """按 first-occurrence 去重，返回保留索引 list。

    朴素 O(N²) 实现就够；真生产用 LSH 加速到 O(N) 期望，但本关只验证正确性。
    """
    # TODO(student):
    #   sigs = [minhash_signature(t, num_perm, n_gram) for t in texts]
    #   keep_indices = []
    #   keep_sigs = []
    #   for i, sig in enumerate(sigs):
    #       is_dup = any(jaccard_estimate(sig, k) >= threshold for k in keep_sigs)
    #       if not is_dup:
    #           keep_indices.append(i)
    #           keep_sigs.append(sig)
    #   return keep_indices
    raise NotImplementedError("L19: implement dedup")
