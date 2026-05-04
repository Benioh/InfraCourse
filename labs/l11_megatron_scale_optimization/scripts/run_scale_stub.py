from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l11_megatron_scale_optimization"
LAB_DIR = Path(__file__).resolve().parents[1]


def estimate(tp: int, pp: int, recompute: bool, gpus: int) -> dict:
    model_gb = 18.0 / max(tp, 1)
    activation_gb = 10.0 / max(pp, 1) * (0.45 if recompute else 1.0)
    optimizer_gb = 28.0 / max(gpus // max(tp * pp, 1), 1)
    comm_penalty = 1.0 + 0.08 * (tp - 1) + 0.05 * (pp - 1)
    compute_penalty = 1.22 if recompute else 1.0
    tokens_per_sec = 42000 * gpus / comm_penalty / compute_penalty
    memory = model_gb + activation_gb + optimizer_gb
    mfu = min(0.62, 0.38 + 0.03 * gpus - 0.02 * (tp - 1) - (0.04 if recompute else 0.0))
    return {
        "tp": tp,
        "pp": pp,
        "recompute": recompute,
        "gpus": gpus,
        "estimated_peak_memory_gb": round(memory, 2),
        "estimated_tokens_per_sec": round(tokens_per_sec, 1),
        "mfu_estimate": round(mfu, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Megatron 扩展配置估算实验")
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/4090_debug.yaml")
    parser.add_argument("--mode", default="scaling")
    args = parser.parse_args()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **config},
    )

    scenarios = [
        estimate(1, 1, False, 1),
        estimate(
            int(config.get("tp", 1)),
            int(config.get("pp", 1)),
            bool(config.get("recompute", False)),
            8,
        ),
        estimate(4, 2, True, 8),
    ]
    write_json(run_dir / "artifacts" / "scale_table.json", scenarios)
    for step, row in enumerate(scenarios, 1):
        append_jsonl(
            run_dir / "metrics.jsonl",
            {"timestamp": utc_now(), "step": step, "metric_type": "train", **row},
        )
    write_text(
        run_dir / "train.log",
        "\n".join(json.dumps(x, ensure_ascii=False) for x in scenarios) + "\n",
    )
    best = min(scenarios, key=lambda x: x["estimated_peak_memory_gb"])
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
比较 Megatron TP/PP/recompute 对显存、吞吐和 MFU 的影响。

## 2. 环境与配置
- 配置：`{args.config}`
- 模式：{args.mode}

## 3. 预测
开启 recompute 会降低激活显存，但会增加计算成本；TP 增加后通信成本上升。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
最佳显存配置：{best}

## 6. 诊断
扩展优化必须同时看显存、吞吐、MFU 和 checkpoint I/O，不能只追求单一指标。

## 7. Debug Ticket
建议练习 `mgt_recompute_tradeoff_007`。

## 8. PR Review
如果 patch 只改 TP 而不改 checkpoint/data parallel 语义，需要拒绝。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
把估算表替换为真实 Megatron logs，并保留同样的 metrics 字段。

## 11. 下一步
在 L04 可运行命令上逐步 sweep micro batch、TP 和 recompute。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
