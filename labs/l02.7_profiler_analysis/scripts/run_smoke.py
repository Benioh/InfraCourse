"""L02.7 Smoke runner: 生成 profiler 分析的证据 artifacts。"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch" / "reference"))
from profiler_analyzer import (
    parse_op_summary,
    classify_bottleneck,
    detect_gpu_idle_segments,
    memory_breakdown,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/4090_debug.yaml")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()

    run_id = f"run_{int(time.time())}"
    out_dir = Path(f"runs/mini_infra/l02.7_profiler_analysis/{run_id}")
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "command.sh").write_text(
        f"python {__file__} --config {args.config} --mode {args.mode}\n"
    )

    sample_events = [
        {"name": "aten::mm", "cpu_time_us": 200, "cuda_time_us": 5000, "calls": 48, "input_shapes": [[4096, 4096]]},
        {"name": "aten::addmm", "cpu_time_us": 150, "cuda_time_us": 3000, "calls": 24, "input_shapes": [[4096]]},
        {"name": "aten::layer_norm", "cpu_time_us": 80, "cuda_time_us": 1200, "calls": 24, "input_shapes": [[1, 2048, 4096]]},
        {"name": "aten::softmax", "cpu_time_us": 30, "cuda_time_us": 600, "calls": 24, "input_shapes": [[1, 32, 2048, 2048]]},
        {"name": "aten::add", "cpu_time_us": 50, "cuda_time_us": 400, "calls": 100, "input_shapes": [[4096]]},
    ]

    summary = parse_op_summary(sample_events, top_k=5)
    (artifacts_dir / "profiler_summary.json").write_text(json.dumps(summary, indent=2))

    stats = {
        "gpu_utilization": 0.88,
        "top_op_type": "matmul",
        "avg_kernel_gap_us": 8.0,
        "comm_pct": 5.0,
        "data_wait_pct": 2.0,
    }
    bottleneck = classify_bottleneck(stats)
    (artifacts_dir / "bottleneck_analysis.json").write_text(json.dumps(bottleneck, indent=2))

    timeline = [
        {"start_us": 0, "end_us": 500, "name": "matmul_fwd"},
        {"start_us": 510, "end_us": 900, "name": "layernorm_fwd"},
        {"start_us": 1500, "end_us": 2000, "name": "matmul_bwd"},
    ]
    idle = detect_gpu_idle_segments(timeline, threshold_us=100.0)
    (artifacts_dir / "gpu_idle_segments.json").write_text(json.dumps(idle, indent=2))

    allocs = [
        {"size_bytes": 500_000_000, "category": "parameter", "is_live": True},
        {"size_bytes": 500_000_000, "category": "gradient", "is_live": True},
        {"size_bytes": 1_000_000_000, "category": "optimizer_state", "is_live": True},
        {"size_bytes": 2_000_000_000, "category": "activation", "is_live": True},
    ]
    mem = memory_breakdown(allocs)
    (artifacts_dir / "memory_breakdown.json").write_text(json.dumps(mem, indent=2))

    metrics = {
        "lab": "l02.7_profiler_analysis",
        "mode": args.mode,
        "top_op": summary[0]["name"] if summary else "none",
        "bottleneck": bottleneck["bottleneck"],
        "gpu_idle_segments": len(idle),
        "largest_mem_category": mem["largest_category"],
        "timestamp": time.time(),
    }
    (out_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")

    report = [
        "# L02.7 Smoke Report",
        "",
        f"- Mode: {args.mode}",
        f"- Top op: {summary[0]['name']} ({summary[0]['pct']:.1f}%)" if summary else "- No ops",
        f"- Bottleneck: {bottleneck['bottleneck']} ({bottleneck['confidence']})",
        f"- GPU idle segments: {len(idle)}",
        f"- Largest memory category: {mem['largest_category']}",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")

    print(f"Smoke complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
