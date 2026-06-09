"""L01.5 Smoke runner: 生成 GPU 执行模型的证据 artifacts。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch" / "reference"))
from gpu_exec_probe import (
    describe_dispatch_chain,
    classify_memory_hierarchy,
    classify_op_bottleneck,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/4090_debug.yaml")
    parser.add_argument("--mode", default="smoke", choices=["smoke", "cluster-smoke"])
    args = parser.parse_args()

    run_id = f"run_{int(time.time())}"
    out_dir = Path(f"runs/mini_infra/l01.5_gpu_execution_model/{run_id}")
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "command.sh").write_text(
        f"python {__file__} --config {args.config} --mode {args.mode}\n"
    )

    chain = describe_dispatch_chain("matmul")
    (artifacts_dir / "dispatch_chain.json").write_text(
        json.dumps({"op": "matmul", "stages": chain}, indent=2)
    )

    hierarchy = classify_memory_hierarchy()
    (artifacts_dir / "hardware_spec.json").write_text(
        json.dumps({"memory_hierarchy": hierarchy}, indent=2)
    )

    matmul_bottleneck = classify_op_bottleneck("matmul", 4096, 4096, 4096)
    elem_bottleneck = classify_op_bottleneck("elementwise", 4096, 4096, 0)
    (artifacts_dir / "bottleneck_analysis.json").write_text(
        json.dumps({"matmul_4096": matmul_bottleneck, "elementwise_4096": elem_bottleneck}, indent=2)
    )

    try:
        import torch
        if torch.cuda.is_available():
            from gpu_exec_probe import measure_async_gap
            async_result = measure_async_gap(2048, "cuda")
            (artifacts_dir / "async_timing.json").write_text(
                json.dumps(async_result, indent=2)
            )
    except Exception as e:
        (artifacts_dir / "async_timing.json").write_text(
            json.dumps({"skipped": True, "reason": str(e)}, indent=2)
        )

    metrics = {
        "lab": "l01.5_gpu_execution_model",
        "mode": args.mode,
        "dispatch_stages": len(chain),
        "memory_levels": len(hierarchy),
        "timestamp": time.time(),
    }
    (out_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")

    report_lines = [
        "# L01.5 Smoke Report",
        "",
        f"- Mode: {args.mode}",
        f"- Dispatch chain: {' → '.join(chain)}",
        f"- Memory hierarchy levels: {len(hierarchy)}",
        f"- MatMul(4096) bottleneck: {matmul_bottleneck['bottleneck']}",
        f"- Elementwise(4096) bottleneck: {elem_bottleneck['bottleneck']}",
        "",
        "## Artifacts",
        "",
    ]
    for f in sorted(artifacts_dir.iterdir()):
        report_lines.append(f"- {f.name}")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(f"Smoke complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
