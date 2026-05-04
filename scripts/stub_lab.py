from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用教学验证运行器：仅用于没有专属 runner 的扩展任务"
    )
    parser.add_argument("--mission", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument(
        "--metric-type", choices=["train", "serve", "rl", "data"], default="train"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="validation")
    parser.add_argument("--title", default="教学验证运行")
    args = parser.parse_args()

    run_dir = prepare_run_dir(args.mission, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": args.mission, "framework": args.framework, "mode": args.mode},
    )
    log_name = (
        "serve.log"
        if args.metric_type == "serve"
        else "rl.log" if args.metric_type == "rl" else "train.log"
    )
    write_text(
        run_dir / log_name,
        f"[{utc_now()}] {args.title} 完成：framework={args.framework}, mode={args.mode}\n",
    )
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": args.metric_type,
            "status": "validation_completed",
            "framework": args.framework,
        },
    )
    write_text(
        run_dir / "report.md",
        f"# Mission Report：{args.mission}\n\n"
        "## 1. 目标\n\n"
        f"完成 {args.framework} 的教学验证运行，并记录命令、配置、日志和指标。\n\n"
        "## 2. 环境与配置\n\n"
        f"- 模式：{args.mode}\n- 框架：{args.framework}\n\n"
        "## 3. 预测\n\n最可能的风险是把配置验证误当成真实训练或真实 serving。\n\n"
        "## 4. 运行命令\n\n见 `command.sh`。\n\n"
        "## 5. 结果\n\n已生成 metrics、日志和中文报告。\n\n"
        "## 6. 诊断\n\n该 runner 只用于教学验证；如果 mission 有专属脚本，应优先使用专属脚本。\n\n"
        "## 7. Debug Ticket\n\n选择本 mission README 中列出的 ticket 做最小复现。\n\n"
        "## 8. PR Review\n\n不要用 validation_completed 掩盖真实框架失败。\n\n"
        "## 9. 我原来误解了什么\n\n"
        "## 10. 如果迁移到 8×H200\n\n替换为真实框架命令，并保留相同 run contract。\n\n"
        "## 11. 下一步\n\n运行专属 smoke 或真实集群命令。\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
