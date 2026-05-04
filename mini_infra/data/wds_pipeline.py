"""
WebDataset 风格 pipeline 模拟入口（L06.3 教学）。

教学目的：
    把 "tar shard → ShardList → split_by_node → split_by_worker → decode → batch"
    流水线的端到端骨架接成可一键跑的 smoke。学员能看到：
        - throughput 与 (workers × prefetch) 的关系
        - p99 batch latency 怎么被 prefetch 吸收
        - detshuffle 在固定 seed 下两次跑出完全相同的顺序
        - corrupt shard 触发的跳过 + cursor++ 路径
    再去 lab 用真实 webdataset 库跑 5-50GB tar 集。

真实框架对照：
    - github_repo/webdataset/webdataset/dataset.py
        真实 WebDataset Pipeline；本文件只算 throughput 估算与拓扑。
    - github_repo/webdataset/webdataset/shardlists.py
        真实 ShardList 与 detshuffle 实现（用 hashlib + seed）。

简化掉的复杂度：
    - 不真读 tar；throughput 用启发式公式
    - 不真做 multi-worker；workers 只参与 throughput 公式
    - 不调真实 decoder（jpg/json/tar.gz）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_infra.data.minhash_dedup import find_near_duplicates
from mini_infra.data.shard_resume import detshuffle_order, recovery_plan
from mini_infra.observability.io import command_snapshot, mini_run_dir, write_json


SAMPLE_TEXTS = [
    "a red fox jumps over the small log",
    "a red fox jumps over a small log",
    "distributed training needs deterministic data shards",
    "serving systems measure ttft and inter token latency",
    "distributed training needs deterministic data shard order",
]


def simulate_pipeline(
    shards: list[str] | None = None,
    workers: int = 4,
    prefetch: int = 8,
    shuffle: str = "detshuffle",
    corrupt_count: int = 1,
) -> dict[str, object]:
    shard_list = shards or [f"shard-{index:04d}.tar" for index in range(8)]
    ordered = detshuffle_order(shard_list, seed=1234) if shuffle == "detshuffle" else shard_list
    corrupt = set(ordered[:corrupt_count])
    recovery = recovery_plan(ordered, corrupt_shards=corrupt)
    dedup = find_near_duplicates(SAMPLE_TEXTS, threshold=0.55)
    throughput = round(180.0 * max(workers, 1) * (1 + min(prefetch, 8) / 16), 3)
    return {
        "workers": workers,
        "prefetch": prefetch,
        "shuffle": shuffle,
        "wds_throughput_mbs": throughput,
        "p99_batch_latency_ms": round(1200 / max(prefetch, 1) + 12 / max(workers, 1), 3),
        "dedup_ratio": dedup["dedup_ratio"],
        "detshuffle_match": ordered == detshuffle_order(shard_list, seed=1234),
        **recovery,
        "near_duplicates": dedup["duplicate_pairs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra WebDataset-style pipeline smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--shards", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=8)
    args = parser.parse_args()
    shards = [Path(path).name for path in args.shards] if args.shards else None
    payload = simulate_pipeline(shards=shards, workers=args.workers, prefetch=args.prefetch)
    if args.run_id:
        run_dir = mini_run_dir("data_engineering", args.run_id)
        command_snapshot(run_dir / "command.sh")
        write_json(run_dir / "artifacts" / "wds_throughput.json", payload)
        write_json(
            run_dir / "artifacts" / "dedup_report.json",
            {"dedup_ratio": payload["dedup_ratio"], "pairs": payload["near_duplicates"]},
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
