"""
GPU kernel microbench 入口（L01.7 教学）。

教学目的：
    把 memory_model + triton_softmax 拼成"一键预算 + 一键报告"的最小入口。
    学员跑 `python -m mini_infra.gpu.microbench --run-id demo` 就能拿到
    BLOCK_SIZE × seq 的完整 sweep，并落到 runs/.../bench_softmax.json。

真实框架对照：
    - github_repo/triton/python/tutorials/02-fused-softmax.py 的 benchmark
      入口；本文件是它的 CPU-safe 同构骨架，不依赖 GPU 即可运行。

诚实边界：
    本 sweep 完全是 roofline 估算（validation_only=True）。报告里写性能
    结论时必须以 ncu 或 torch.profiler 实测覆盖；不能直接拿 CPU sim 当
    Triton 性能。
"""
from __future__ import annotations

import argparse
import json

from mini_infra.gpu.triton_softmax import simulate_softmax_kernel
from mini_infra.observability.io import command_snapshot, mini_run_dir, write_json


def run_sweep(device: str = "rtx4090") -> dict[str, object]:
    rows = []
    for seq_len in (1024, 4096, 16384):
        for block_size in (512, 1024, 2048):
            rows.append(
                simulate_softmax_kernel(
                    batch=8,
                    seq_len=seq_len,
                    block_size=block_size,
                    device=device,
                )
            )
    best = max(rows, key=lambda row: float(row["bandwidth_gbs"]))
    return {
        "benchmark": "fused_softmax_roofline_sim",
        "device": device,
        "rows": rows,
        "best": best,
        "validation_only": True,
        "note": "CPU-safe model; replace with Triton/Nsight measurements for performance claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra GPU kernel microbench")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="rtx4090")
    args = parser.parse_args()
    payload = run_sweep(args.device)
    if args.run_id:
        run_dir = mini_run_dir("gpu_kernel", args.run_id)
        command_snapshot(run_dir / "command.sh")
        write_json(run_dir / "artifacts" / "bench_softmax.json", payload)
        write_json(
            run_dir / "artifacts" / "profile_kineto.json",
            {
                "profiler": "torch.profiler.kineto-compatible summary",
                "events": [payload["best"]],
            },
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
