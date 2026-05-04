from __future__ import annotations

import argparse
import json
import subprocess
import sys

from mini_infra.observability.io import ROOT


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniInfra end-to-end smoke")
    parser.add_argument("--run-id", default="smoke")
    args = parser.parse_args()
    python = sys.executable
    train_run = f"{args.run_id}_train"
    serving_run = f"{args.run_id}_serving"
    rl_run = f"{args.run_id}_rl"
    run([python, "-m", "mini_infra.data.toy_data"])
    run(
        [
            python,
            "-m",
            "mini_infra.data.indexed_dataset",
            "--output-prefix",
            f"runs/mini_infra/data/{args.run_id}/toy_text_document",
        ]
    )
    run(
        [
            python,
            "-m",
            "mini_infra.training.train_tiny",
            "--backend",
            "simulated",
            "--run-id",
            train_run,
        ]
    )
    checkpoint = ROOT / "runs" / "mini_infra" / "train" / train_run / "checkpoint"
    run(
        [
            python,
            "-m",
            "mini_infra.checkpointing.convert_hf",
            "--checkpoint",
            str(checkpoint),
            "--output",
            f"runs/mini_infra/hf/{args.run_id}",
        ]
    )
    run([python, "-m", "mini_infra.serving.benchmark", "--run-id", serving_run])
    run([python, "-m", "mini_infra.serving.prefix_cache"])
    run([python, "-m", "mini_infra.serving.pd_simulator"])
    run([python, "-m", "mini_infra.rl.rollout", "--run-id", rl_run])
    run([python, "-m", "mini_infra.rl.actor_rollout_plan"])
    run(
        [
            python,
            "-m",
            "mini_infra.reports.build_delivery",
            "--output",
            f"runs/mini_infra/delivery/{args.run_id}",
        ]
    )
    run(
        [
            python,
            "-m",
            "mini_infra.real_stack_smoke",
            "--output",
            f"runs/mini_infra/real_stack/{args.run_id}.json",
        ]
    )
    print(
        json.dumps(
            {"status": "ok", "run_id": args.run_id}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
