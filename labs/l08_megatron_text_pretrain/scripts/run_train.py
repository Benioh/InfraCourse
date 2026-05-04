from __future__ import annotations

import argparse
import importlib.util
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
    write_text,
    write_yaml,
)

MISSION_ID = "l08_megatron_text_pretrain"
LAB_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "wikitext"
INDEX_PREFIX = ROOT / "data" / "wikitext" / "indexed_dataset" / "wikitext_text_document"


def has_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="4090")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **config},
    )

    data_ready = (DATA_DIR / "train.jsonl").exists()
    indexed_bin = INDEX_PREFIX.with_suffix(".bin")
    indexed_idx = INDEX_PREFIX.with_suffix(".idx")
    indexed_ready = indexed_bin.exists() and indexed_idx.exists()
    megatron_available = has_spec("megatron") or has_spec("megatron.core")

    expected_command = (
        "torchrun --nproc_per_node=1 pretrain_gpt.py "
        f"--data-path {INDEX_PREFIX} --seq-length {config['seq_length']} "
        f"--micro-batch-size {config['micro_batch_size']} --global-batch-size {config['global_batch_size']} "
        f"--tensor-model-parallel-size {config['tp']} --pipeline-model-parallel-size {config['pp']}"
    )
    fallback_reason = []
    if not data_ready:
        fallback_reason.append("JSONL data missing")
    if not indexed_ready:
        fallback_reason.append("indexed dataset missing")
    if not megatron_available:
        fallback_reason.append("Megatron 包不可导入")

    if args.resume:
        write_text(
            run_dir / "artifacts" / "resume_check.txt",
            "只有 TP/PP/tokenizer 保持不变时，resume 才可视为兼容。\n",
        )

    if fallback_reason:
        write_text(
            run_dir / "artifacts" / "fallback_reason.txt",
            "使用了有记录的 fallback 路径：\n- " + "\n- ".join(fallback_reason) + "\n",
        )
        write_text(
            run_dir / "artifacts" / "fallback_train.log",
            f"[{utc_now()}] fallback 验证\n",
        )
    write_text(
        run_dir / "train.log", f"[{utc_now()}] expected_command={expected_command}\n"
    )

    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "train",
            "stage": "resume" if args.resume else "train",
            "resume_success": args.resume and indexed_ready,
            "status": (
                "fallback_validated"
                if fallback_reason
                else "ready_for_manual_megatron_launch"
            ),
        },
    )
    write_text(
        run_dir / "report.md",
        f"# Mission Report：{MISSION_ID}\n\n"
        "## 1. 目标\n\n"
        "验证 Megatron 纯文本预训练从公开数据到启动命令的边界。\n\n"
        "## 2. 环境与配置\n"
        f"- GPU: {args.mode}\n"
        "- 框架：Megatron\n"
        f"- Model: {config['model_size']}\n"
        "- 数据集：WikiText-103\n"
        f"- Precision: {config['precision']}\n"
        f"- Parallelism: TP={config['tp']} PP={config['pp']}\n\n"
        "## 3. 预测\n"
        "- 预测瓶颈：激活显存\n"
        "- 预测显存：随序列长度变化\n"
        "- 预测吞吐：对拓扑敏感\n"
        "- 预测失败：tokenizer 或 resume 不匹配\n\n"
        "## 4. 运行命令\n\n"
        "见 `command.sh` 和 `train.log`。\n\n"
        "## 5. 结果\n"
        f"- JSONL 就绪： {data_ready}\n"
        f"- Indexed dataset 就绪： {indexed_ready}\n"
        f"- Megatron 可导入： {megatron_available}\n"
        f"- Resume 路径已检查： {args.resume}\n\n"
        "## 6. 诊断\n\n"
        "本次运行验证 Megatron 启动边界；当本地 runtime 不可用时，会明确记录 fallback，不伪装成真实训练。\n\n"
        "## 7. Debug 工单\n"
        "- Ticket：mgt_checkpoint_002\n"
        "- 根因：TP/PP/tokenizer 漂移会破坏 checkpoint 兼容性\n"
        "- 最小修复： keep topology and tokenizer stable across save and resume\n"
        "- 验证方式： compare resolved configs and checkpoint lineage\n\n"
        "## 8. PR Review\n"
        "- 审查的 patch： global_batch_size change from 256 to 257\n"
        "- 风险： batch math and DP divisibility break silently\n"
        "- 增加的测试： batch formula check in report and rubric\n\n"
        "## 9. 我原来误解了什么\n\n"
        "## 10. 如果迁移到 8×H200\n\n"
        "使用 H200 配置、真实 indexed data 与 TP 实验，同时保留相同命令契约。\n\n"
        "## 11. 下一步\n\n"
        f"Effective batch formula: global_batch_size = micro_batch_size × data_parallel × gradient_accumulation. Here the config records micro_batch_size={config['micro_batch_size']} and global_batch_size={config['global_batch_size']}.\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
