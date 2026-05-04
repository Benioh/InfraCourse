from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_infra.checkpointing.metadata import write_manifest
from mini_infra.model.tiny_transformer import TinyModelConfig
from mini_infra.observability.io import (
    command_snapshot,
    mini_run_dir,
    write_json,
    write_text,
    write_yaml,
)
from mini_infra.training.trainer import MiniTrainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra tiny LM training")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--backend", choices=["simulated", "torch"], default="simulated"
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    run_dir = mini_run_dir("train", args.run_id)
    command_snapshot(
        run_dir / "command.sh",
        ["python", "-m", "mini_infra.training.train_tiny", "--backend", args.backend],
    )
    model_config = TinyModelConfig(max_seq_len=max(args.seq_len, 128))
    train_config = TrainConfig(
        backend=args.backend,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"model": model_config.to_dict(), "train": train_config.to_dict()},
    )
    summary = MiniTrainer(model_config, train_config, run_dir).run()
    manifest = write_manifest(
        Path(summary["checkpoint_dir"]),
        "training",
        model_config.to_dict(),
        {"final_loss": summary["final_loss"], "backend": args.backend},
    )
    write_json(
        run_dir / "artifacts" / "training_summary.json",
        {"summary": summary, "manifest": manifest},
    )
    write_text(
        run_dir / "report.md",
        f"""# MiniInfra Training Report

## 目标
训练或模拟训练一个 tiny transformer，并产出 checkpoint manifest。

## 结果
- backend：{args.backend}
- final_loss：{summary['final_loss']}
- checkpoint：`{summary['checkpoint_dir']}`

## 迁移判断
模拟 backend 只验证训练证据链；torch backend 才能支撑真实 loss/梯度路径判断。
""",
    )
    print(
        json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
