"""
MinHash 近重复检测最小同构（L19 教学）。

教学目的：
    把 "MinHash + LSH 求近重复" 从论文公式变成可跑的函数。学员在 n16
    notebook 里能：
        - 看 jaccard 与 minhash signature similarity 的关系（无偏估计）
        - 调 num_perm 看精度-速度权衡
        - 调 threshold 看假阳性数变化
        - 在合成对（如"a red fox jumps over the small log" vs "...a small log"）
          上验证 jaccard ≈ minhash_similarity

真实框架对照：
    - github_repo/datasketch/datasketch/minhash.py
        真实 MinHash 用 numpy + 多组哈希函数；本文件用 sha1 替代。
    - github_repo/datasketch/datasketch/lsh.py
        LSH band/row 切分以 O(N) 找近邻；本文件直接 N^2 比对。

简化掉的复杂度：
    - 没有 LSH 分桶（在 N >> 1000 时太慢）
    - 用 sha1 替代真实 universal hash family
    - shingle width=3 固定；真实可调
"""
from __future__ import annotations

import hashlib
from itertools import combinations


def shingles(text: str, width: int = 3) -> set[str]:
    """把文本按 token 划成 width-gram 集合。

    width=3 是常见选择：太小（1-2）假阳性多，太大（6+）召回率低。
    真实使用前应做 lowercase / unicode normalize / strip punctuation。
    """
    tokens = text.lower().split()
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:12], 16)


def minhash_signature(text: str, num_perm: int = 64, width: int = 3) -> list[int]:
    grams = shingles(text, width)
    if not grams:
        return [0 for _ in range(num_perm)]
    return [min(stable_hash(gram, seed) for gram in grams) for seed in range(num_perm)]


def signature_similarity(lhs: list[int], rhs: list[int]) -> float:
    if not lhs or len(lhs) != len(rhs):
        return 0.0
    return round(
        sum(1 for left, right in zip(lhs, rhs, strict=False) if left == right) / len(lhs), 6
    )


def jaccard(lhs: str, rhs: str, width: int = 3) -> float:
    left = shingles(lhs, width)
    right = shingles(rhs, width)
    if not left and not right:
        return 1.0
    return round(len(left & right) / max(len(left | right), 1), 6)


def find_near_duplicates(texts: list[str], threshold: float = 0.8) -> dict[str, object]:
    signatures = [minhash_signature(text) for text in texts]
    pairs = []
    for left, right in combinations(range(len(texts)), 2):
        estimate = signature_similarity(signatures[left], signatures[right])
        if estimate >= threshold:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "minhash_similarity": estimate,
                    "jaccard": jaccard(texts[left], texts[right]),
                }
            )
    return {
        "num_docs": len(texts),
        "threshold": threshold,
        "duplicate_pairs": pairs,
        "dedup_ratio": round(len({pair["right"] for pair in pairs}) / max(len(texts), 1), 6),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MiniHash dedup smoke")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--shards", nargs="*", default=[])
    args = parser.parse_args()
    docs = args.shards or [
        "a red fox jumps over the small log",
        "a red fox jumps over a small log",
        "completely different document",
    ]
    print(
        json.dumps(
            find_near_duplicates(docs, threshold=args.threshold), ensure_ascii=False, indent=2
        )
    )
