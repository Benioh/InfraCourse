from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l31_rollout_only_smoke"


PROMPTS = [
    {
        "id": "gsm8k_toy_001",
        "prompt": "If Alice has 3 apples and buys 4 more, how many apples does she have?",
        "answer": "7",
    },
    {
        "id": "gsm8k_toy_002",
        "prompt": "A box has 5 red balls and 6 blue balls. How many balls are there?",
        "answer": "11",
    },
    {
        "id": "gsm8k_toy_003",
        "prompt": "Tom reads 2 pages per minute for 8 minutes. How many pages?",
        "answer": "16",
    },
    {
        "id": "gsm8k_toy_004",
        "prompt": "There are 9 birds, 3 fly away. How many remain?",
        "answer": "6",
    },
]


def mock_response(prompt: dict[str, str], max_new_tokens: int) -> dict[str, Any]:
    start = time.perf_counter()
    time.sleep(0.005 + random.random() * 0.003)
    latency_ms = (time.perf_counter() - start) * 1000
    response = f"We solve it step by step. Final answer: {prompt['answer']}"
    return {
        "id": prompt["id"],
        "prompt": prompt["prompt"],
        "response": response,
        "expected_answer": prompt["answer"],
        "ttft_ms": round(latency_ms, 3),
        "latency_ms": round(latency_ms + max_new_tokens * 0.01, 3),
        "output_tokens_est": min(max_new_tokens, len(response.split())),
        "mock_server": True,
        "reward_input_ready": "Final answer:" in response,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {
            "mission": MISSION_ID,
            "mode": args.mode,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "actor_update": False,
            "weight_sync": False,
        },
    )

    selected = [PROMPTS[index % len(PROMPTS)] for index in range(args.request_count)]
    started = time.perf_counter()
    rows = [mock_response(prompt, args.max_new_tokens) for prompt in selected]
    elapsed = max(time.perf_counter() - started, 1e-6)
    write_jsonl(run_dir / "artifacts" / "rollouts.jsonl", rows)

    avg_ttft = statistics.mean(row["ttft_ms"] for row in rows)
    avg_output_tokens = statistics.mean(row["output_tokens_est"] for row in rows)
    rollouts_per_sec = len(rows) / elapsed
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "rollout_only",
            "request_count": len(rows),
            "rollouts_per_sec": round(rollouts_per_sec, 3),
            "avg_ttft_ms": round(avg_ttft, 3),
            "avg_output_tokens": round(avg_output_tokens, 3),
            "mock_server": True,
            "actor_update": False,
            "weight_sync": False,
        },
    )
    write_text(
        run_dir / "rl.log",
        f"[{utc_now()}] rollout-only smoke 完成：requests={len(rows)} mock_server=True\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
只验证 rollout prompt/response/latency/schema；本关不接 Megatron actor、Ray 训练循环或 weight sync。

## 2. 源码调用链
`python labs/l31_rollout_only_smoke/scripts/run_rollout_only.py` → `artifacts/rollouts.jsonl` → `metrics.jsonl` → `report.md`。

## 3. 实验矩阵
| run_id | 只改变的变量 | 关键指标 | 结论 |
|---|---|---|---|
| {run_dir.name} | request_count={args.request_count} | rollouts_per_sec={rollouts_per_sec:.3f} | rollout schema 可观测 |
| boundary | mock_server=True | weight_sync=False | 不证明真实 SGLang 或 actor sync |

## 4. 指标结果
- request_count：{len(rows)}
- rollouts_per_sec：{rollouts_per_sec:.3f}
- avg_ttft_ms：{avg_ttft:.3f}
- avg_output_tokens：{avg_output_tokens:.3f}
- mock_server：True

## 5. Debug Ticket
建议练习 `verl_rollout_slow_003` 或 `slime_rollout_bottleneck_001`；证据路径为 `artifacts/rollouts.jsonl`、`metrics.jsonl` 和 `rl.log`。

## 6. 迁移判断
本关可以迁移 response schema 与 reward-input 检查；不能迁移真实 SGLang 性能、Megatron actor 更新或 weight sync 正确性。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
