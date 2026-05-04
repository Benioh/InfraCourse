from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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

MISSION_ID = "l01_env_conda_cuda"
LAB_DIR = Path(__file__).resolve().parents[1]


def run_python(script: str, output: Path) -> None:
    subprocess.run(
        [sys.executable, str(LAB_DIR / "scripts" / script), "--output", str(output)],
        check=True,
    )


def run_torchrun(log_path: Path, nproc_per_node: int) -> bool:
    script = LAB_DIR / "scripts" / "torchrun_hello.py"
    cmd = (
        [shutil.which("torchrun"), "--nproc_per_node", str(nproc_per_node), str(script)]
        if shutil.which("torchrun")
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc_per_node),
            str(script),
        ]
    )
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as exc:
            handle.write(f"\nlaunch_failed={exc.returncode}\n")
            return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()

    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **config},
    )
    run_python("check_cuda.py", run_dir / "artifacts" / "check_cuda.json")
    run_python("check_torch.py", run_dir / "artifacts" / "check_torch.json")
    run_python("check_nccl.py", run_dir / "artifacts" / "check_nccl.json")
    run_python("collect_env.py", run_dir / "artifacts" / "collect_env.json")
    torchrun_ok = run_torchrun(
        run_dir / "artifacts" / "torchrun_hello.log",
        int(config["torchrun_nproc_per_node"]),
    )

    check_cuda = json.loads(
        (run_dir / "artifacts" / "check_cuda.json").read_text(encoding="utf-8")
    )
    write_text(
        run_dir / "train.log",
        (
            f"[{utc_now()}] Environment validation complete\n"
            f"cuda_available={check_cuda['cuda_available']}\n"
            f"cuda_device_count={check_cuda['cuda_device_count']}\n"
            f"torchrun_ok={torchrun_ok}\n"
        ),
    )
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "step": 0,
            "metric_type": "train",
            "loss": 0.0,
            "tokens_per_sec": 0.0,
            "peak_memory_gb": 0.0,
            "gpu_util": 0.0,
            "torchrun_ok": torchrun_ok,
            "cuda_available": check_cuda["cuda_available"],
        },
    )
    write_text(
        run_dir / "report.md",
        f"# Mission Report：{MISSION_ID}\n\n"
        "## 1. 目标\n\n"
        "验证 Python、CUDA、NCCL 与 torchrun 基础环境。\n\n"
        "## 2. 环境与配置\n"
        f"- GPU: {config['gpu_mode']}\n"
        "- 框架：PyTorch\n"
        "- 模型：无\n"
        "- 数据集：无\n"
        "- 精度：不适用\n"
        "- 并行：本地多进程检查\n\n"
        "## 3. 预测\n"
        "- 预测瓶颈：环境漂移\n"
        "- 预测显存：几乎可以忽略\n"
        "- 预测吞吐：不适用\n"
        "- 预测失败：rank/device 映射不一致\n\n"
        "## 4. 运行命令\n\n"
        "见 `command.sh`。\n\n"
        "## 5. 结果\n"
        f"- CUDA 可见： {check_cuda['cuda_available']}\n"
        f"- 设备数量： {check_cuda['cuda_device_count']}\n"
        f"- torchrun 正常： {torchrun_ok}\n\n"
        "## 6. 诊断\n\n"
        "环境检查已完成，所有证据保存在 `artifacts/`。\n\n"
        "## 7. Debug 工单\n"
        "- Ticket：env_rank_device_mismatch_003\n"
        "- 根因：需要通过 torchrun 输出验证 rank 到 device 的映射\n"
        "- 最小修复： bind device from `LOCAL_RANK`\n"
        "- 验证方式： inspect `artifacts/torchrun_hello.log`\n\n"
        "## 8. PR Review\n"
        "- 审查的 patch： n/a\n"
        "- 风险： wrong interpreter or wrong launch mapping\n"
        "- 增加的测试： environment smoke itself\n\n"
        "## 9. 我原来误解了什么\n\n"
        "## 10. 如果迁移到 8×H200\n\n"
        "把 `--nproc_per_node` 提到 8，并确认所有 rank 都发现预期设备。\n\n"
        "## 11. 下一步\n\n"
        "进入 L01，开始 profile 一个真实训练循环。\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
