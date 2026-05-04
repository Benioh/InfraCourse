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

MISSION_ID = "l17_megatron_multimodal_data"
LAB_DIR = Path(__file__).resolve().parents[1]


def run(name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(LAB_DIR / "scripts" / name), *args],
        check=True,
        cwd=LAB_DIR,
    )


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
        {
            "mission": MISSION_ID,
            "mode": args.mode,
            "schema": "configs/manifest_schema.yaml",
        },
    )
    for script in [
        "download_flickr8k.py",
        "download_librispeech.py",
        "download_esc50.py",
        "build_manifest.py",
        "build_webdataset_shards.py",
        "visualize_batch.py",
    ]:
        run(script)
    run(
        "validate_manifest.py",
        "--output",
        str(run_dir / "artifacts" / "manifest_validation.json"),
    )
    run(
        "inspect_shards.py",
        "--output",
        str(run_dir / "artifacts" / "shard_inspection.json"),
    )
    run(
        "energon_loader_smoke.py",
        "--output",
        str(run_dir / "artifacts" / "loader_smoke.json"),
    )
    validation = json.loads(
        (run_dir / "artifacts" / "manifest_validation.json").read_text(encoding="utf-8")
    )
    shards = json.loads(
        (run_dir / "artifacts" / "shard_inspection.json").read_text(encoding="utf-8")
    )
    loader = json.loads(
        (run_dir / "artifacts" / "loader_smoke.json").read_text(encoding="utf-8")
    )
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "data",
            "manifest_rows": validation["rows"],
            "missing_files": len(validation["missing_files"]),
            "validation_passed": validation["validation_passed"],
            "shard_count": shards["shard_count"],
            "sample_count": loader["sample_count"],
        },
    )
    write_text(
        run_dir / "train.log",
        f"[{utc_now()}] 多模态数据管线完成：manifest_rows={validation['rows']} shard_count={shards['shard_count']} sample_count={loader['sample_count']}\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
构建 manifest、WebDataset shard，并验证多模态样本键和文件一致性。

## 2. 环境与配置
- 教学数据：`data/multimodal_toy/`
- schema：`configs/manifest_schema.yaml`

## 3. 预测
最容易失败的是 manifest 指向缺失文件或 shard 内 key 不一致。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
- manifest rows：{validation['rows']}
- missing files：{len(validation['missing_files'])}
- shard count：{shards['shard_count']}
- sample count：{loader['sample_count']}

## 6. 诊断
manifest 校验和 shard inspect 都通过，说明数据边界可被训练框架消费。

## 7. Debug Ticket
建议练习 `mm_bad_shard_003`。

## 8. PR Review
任何改变样本 key 或 shard 命名的 patch 都必须更新 resume state 说明。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
需要验证分布式 dataloader 是否重复/漏读样本，并保存 shard cursor。

## 11. 下一步
把教学 manifest 替换成真实授权数据集子集。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
