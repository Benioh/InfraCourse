from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mini_infra.observability.io import ROOT, write_json


def commands(run_id: str) -> list[list[str]]:
    return [
        ["-m", "mini_infra.gpu.microbench", "--run-id", run_id],
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            run_id,
            "--tp",
            "2",
            "--pp",
            "2",
            "--distributed-optimizer",
        ],
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            f"{run_id}_cp",
            "--cp",
            "2",
            "--seq",
            "16384",
            "--yarn-scale",
            "1.0",
        ],
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            f"{run_id}_moe",
            "--moe-experts",
            "8",
            "--moe-topk",
            "2",
            "--ep",
            "2",
        ],
        ["-m", "mini_infra.data.wds_pipeline", "--run-id", run_id],
        ["-m", "mini_infra.vllm.run_engine", "--run-id", run_id, "--quant", "awq"],
        ["-m", "mini_infra.vllm.run_engine", "--run-id", f"{run_id}_spec", "--spec", "ngram"],
        ["-m", "mini_infra.sglang.run_scheduler", "--pd", "--kv-int8", "--run-id", run_id],
        ["-m", "mini_infra.slime.train", "--steps", "2"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-project-shaped MiniInfra stack smoke"
    )
    parser.add_argument("--run-id", default="real_stack")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or f"runs/mini_infra/real_stack/{args.run_id}.json"
    results = []
    for command in commands(args.run_id):
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "command": "python " + " ".join(command),
                "stdout": completed.stdout.strip(),
            }
        )
    write_json(Path(output), results)
    print(
        json.dumps(
            {"status": "ok", "output": output, "commands": len(results)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
