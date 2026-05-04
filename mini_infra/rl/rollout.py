from __future__ import annotations

import argparse
import json
import time
from typing import Any

from mini_infra.data.toy_data import SAMPLES
from mini_infra.observability.io import (
    append_jsonl,
    command_snapshot,
    mini_run_dir,
    utc_now,
    write_json,
    write_text,
)
from mini_infra.rl.reward import score


def run_rollout(max_new_tokens: int = 64) -> list[dict[str, Any]]:
    rows = []
    for sample in SAMPLES:
        started = time.perf_counter()
        response = f"We solve the problem. Final answer: {sample['answer']}"
        latency_ms = (time.perf_counter() - started) * 1000 + max_new_tokens * 0.01
        rows.append(
            {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "response": response,
                "target": f"#### {sample['answer']}",
                "latency_ms": round(latency_ms, 3),
                **score(response, f"#### {sample['answer']}"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra rollout-only smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    run_dir = mini_run_dir("rl", args.run_id)
    command_snapshot(run_dir / "command.sh", ["python", "-m", "mini_infra.rl.rollout"])
    rows = run_rollout(args.max_new_tokens)
    reward_mean = sum(row["reward"] for row in rows) / len(rows)
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "rl",
            "reward_mean": reward_mean,
            "rollout_count": len(rows),
        },
    )
    write_json(run_dir / "artifacts" / "rollouts.json", rows)
    write_text(
        run_dir / "report.md",
        f"# MiniInfra RL Rollout\n\n- reward_mean: {reward_mean}\n- rollout_count: {len(rows)}\n",
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "reward_mean": reward_mean,
                "rollout_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
