from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l05_distributed_primitives"
LAB_DIR = Path(__file__).resolve().parents[1]


def run_script(name: str, output: Path, *extra: str) -> dict:
    command = [
        sys.executable,
        str(LAB_DIR / "scripts" / name),
        "--output",
        str(output),
        *extra,
    ]
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, "recommended_torchrun_nproc": 2},
    )

    artifacts = run_dir / "artifacts"
    collectives = run_script("collectives_demo.py", artifacts / "collectives.json")
    ddp = run_script(
        "ddp_toy_train.py", artifacts / "ddp_toy_train.json", "--steps", "30"
    )
    tp = run_script(
        "toy_column_parallel_linear.py", artifacts / "tensor_parallel_linear.json"
    )
    pp = run_script(
        "toy_pipeline_two_stage.py",
        artifacts / "pipeline_parallel.json",
        "--microbatches",
        "4",
    )

    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "train",
            "all_reduce_ms": collectives.get("all_reduce_ms"),
            "ddp_loss_start": ddp.get("loss_start"),
            "ddp_loss_end": ddp.get("loss_end"),
            "tp_max_error": tp.get("max_error"),
            "pipeline_microbatches": pp.get("microbatches"),
            "pipeline_bubble_ratio": pp.get("bubble_ratio"),
        },
    )
    write_text(
        run_dir / "train.log",
        "\n".join(
            [
                f"[{utc_now()}] L02 分布式原语 smoke 完成",
                f"collectives={collectives}",
                f"ddp={ddp}",
                f"tensor_parallel={tp}",
                f"pipeline={pp}",
            ]
        )
        + "\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
验证 collective、DDP、tensor parallel toy、pipeline toy 的最小闭环。

## 2. 环境与配置
- 模式：{args.mode}
- 推荐 torchrun：`torchrun --nproc_per_node=2 ...`

## 3. 预测
分布式最常见风险是 world size/rank/device 不一致导致 hang。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
- all_reduce_ms：{collectives.get('all_reduce_ms')}
- DDP loss：{ddp.get('loss_start')} → {ddp.get('loss_end')}
- TP 最大误差：{tp.get('max_error')}
- Pipeline microbatches：{pp.get('microbatches')}，bubble={pp.get('bubble_ratio')}

## 6. 诊断
DDP loss 下降说明同步训练闭环可用；TP 误差接近 0 说明切分与 gather 语义正确。

## 7. Debug Ticket
建议从 `dist_wrong_world_size_001` 开始练习。

## 8. PR Review
任何改动 collective 顺序或 rank 条件分支的 patch 都必须先做最小 torchrun 验证。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
需要增加 collective size sweep，并记录 NCCL backend 与网络环境。

## 11. 下一步
进入 TorchTitan/Megatron 前，先能解释 all-reduce 在 DDP 中的位置。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
