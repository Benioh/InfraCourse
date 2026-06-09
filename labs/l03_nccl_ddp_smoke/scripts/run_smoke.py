from __future__ import annotations

import argparse
import json
import subprocess
import sys
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

MISSION_ID = "l03_nccl_ddp_smoke"
LAB_DIR = Path(__file__).resolve().parents[1]


def fallback_payload(reason: str) -> dict[str, Any]:
    return {
        "backend": "simulated-fallback",
        "rank": 0,
        "local_rank": 0,
        "world_size": 2,
        "device": "cpu",
        "all_reduce_sum": 3.0,
        "barrier_ok": True,
        "fallback_used": True,
        "note": reason,
    }


def run_torchrun(output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(LAB_DIR / "scripts" / "ddp_hello.py"),
        "--output",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
    except Exception as exc:
        return fallback_payload(f"torchrun 未完成，使用模拟结果：{exc}")
    if not output.exists():
        return fallback_payload("torchrun 结束但 rank0 artifact 未生成")
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
        {"mission": MISSION_ID, "mode": args.mode, "nproc_per_node": 2},
    )

    artifact_path = run_dir / "artifacts" / "ddp_hello.json"
    payload = run_torchrun(artifact_path)
    if not artifact_path.exists():
        write_text(
            artifact_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )

    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "distributed_hello",
            "world_size": payload.get("world_size"),
            "backend": payload.get("backend"),
            "all_reduce_sum": payload.get("all_reduce_sum"),
            "barrier_ok": payload.get("barrier_ok"),
            "fallback_used": payload.get("fallback_used", False),
        },
    )
    write_text(run_dir / "train.log", f"[{utc_now()}] L04 smoke 完成：{payload}\n")
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
验证 2-rank process group、all_reduce 与 barrier 的最小语义；本关不覆盖 TP/PP。

## 2. 源码调用链
`make smoke` → `scripts/run_smoke.py` → `torch.distributed.run` → `scripts/ddp_hello.py` → `artifacts/ddp_hello.json` → `metrics.jsonl`。

## 3. 实验矩阵
| run_id | 只改变的变量 | 关键指标 | 结论 |
|---|---|---|---|
| {run_dir.name} | mode={args.mode} | backend={payload.get('backend')} | 记录 rank/world_size/all_reduce/barrier |
| baseline | fallback 边界 | fallback={payload.get('fallback_used')} | 不把 fallback 写成真实 NCCL |

## 4. 指标结果
- world_size：{payload.get('world_size')}
- backend：{payload.get('backend')}
- all_reduce_sum：{payload.get('all_reduce_sum')}
- barrier_ok：{payload.get('barrier_ok')}
- fallback_used：{payload.get('fallback_used')}

## 5. Debug Ticket
建议练习 `dist_wrong_world_size_001`，证据路径为 `artifacts/ddp_hello.json` 和 `train.log`。

## 6. 迁移判断
本关只验证 rank/process group 边界；进入 L06 后才把 collective 直觉迁移到 TP/PP toy。若 fallback_used=True，本次结果只能作为 validation-only。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
