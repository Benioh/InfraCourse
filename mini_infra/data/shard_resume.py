"""
Shard 容灾与可复现 shuffle（L19 教学）。

教学目的：
    把 "shard 损坏跳过、续读 cursor、detshuffle 可复现" 三件事的最小语义
    具体化。学员能看到：
        - corrupt shard 不会让整个 epoch 死掉（recovered_shards / lost_samples）
        - resume_cursor 能让重启的训练从下一个 shard 开始
        - detshuffle_order 在相同 seed 下两次输出 byte-equal

真实框架对照：
    - github_repo/webdataset/webdataset/handlers.py
        真实 webdataset 的 `warn_and_continue` / `reraise_exception`
        handler；本文件做 set 查找替代。
    - github_repo/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py
        真实 indexed dataset 的 cursor 与 resume；本文件只做最小骨架。

简化掉的复杂度：
    - 不真读 tar；只做拓扑模拟
    - lost_samples 用 corrupt × 1024 估算
    - 不与 multi-worker cursor 协同
"""
from __future__ import annotations

import hashlib


def recovery_plan(
    shards: list[str], corrupt_shards: set[str] | None = None, start_after: str | None = None
) -> dict[str, object]:
    """模拟 shard 级容灾 + 续读。

    corrupt_shards: 已知损坏的 shard 集合（实际由 IO 层 try/except 捕获）
    start_after: resume cursor，None 表示从头开始；指定时跳到这个 shard 之后

    输出含 recovered（成功读的）、skipped（跳过原因）、lost_samples（估算丢量）。
    """
    corrupt = corrupt_shards or set()
    recovered = []
    skipped = []
    started = start_after is None
    for shard in shards:
        if not started:
            skipped.append({"shard": shard, "reason": "before_resume_cursor"})
            if shard == start_after:
                started = True
            continue
        if shard in corrupt:
            skipped.append({"shard": shard, "reason": "corrupt"})
            continue
        recovered.append(shard)
    return {
        "input_shards": shards,
        "resume_cursor": start_after,
        "recovered_shards": len(recovered),
        "lost_samples": len(corrupt) * 1024,
        "recovered": recovered,
        "skipped": skipped,
    }


def detshuffle_order(shards: list[str], seed: int) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16), value

    return sorted(shards, key=key)
