"""L05.7 · 在 mock 时间模型上演练 1F1B schedule，量化 bubble ratio。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB_DIR / "patch"))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l14_pipeline_1f1b"


def _impl():
    try:
        from starter import pp_schedule as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import pp_schedule as mod  # type: ignore[import-not-found]

        return mod, "reference"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    mod, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )
    if "megatron_command" in config:
        write_text(
            run_dir / "artifacts" / "megatron_pp_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["megatron_command"].split())
            + "\n",
        )

    D = int(config["num_stages"])
    N = int(config["num_microbatches"])
    fwd_t = float(config.get("forward_time", 1.0))
    bwd_t = float(config.get("backward_time", 2.0))

    schedule = mod.make_1f1b_schedule(D, N)
    bubble = mod.bubble_count(D)

    # Each stage processes its own timeline; under perfect parallelism
    # makespan = max stage timeline length × per-op time approx
    longest_stage_seconds = max(
        sum(fwd_t if op == "F" else bwd_t for op, _ in tl) for tl in schedule
    )
    ideal_seconds = N * (fwd_t + bwd_t)
    bubble_seconds = longest_stage_seconds - ideal_seconds
    bubble_ratio = bubble_seconds / longest_stage_seconds if longest_stage_seconds else 0

    bubble_eq = config.get("acceptance", {}).get("bubble_eq")
    total_ops_eq = config.get("acceptance", {}).get("total_ops_eq")
    bubble_ratio_max = config.get("acceptance", {}).get("bubble_ratio_max")
    accept_ok = True
    if bubble_eq is not None and bubble != int(bubble_eq):
        accept_ok = False
    total_ops = sum(len(tl) for tl in schedule)
    if total_ops_eq is not None and total_ops != int(total_ops_eq):
        accept_ok = False
    if bubble_ratio_max is not None and bubble_ratio > float(bubble_ratio_max):
        accept_ok = False

    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "pp",
            "stages": D,
            "microbatches": N,
            "bubble_count": bubble,
            "longest_stage_seconds": longest_stage_seconds,
            "bubble_ratio": bubble_ratio,
            "accept": accept_ok,
        },
    )

    write_json(
        run_dir / "artifacts" / "pp_summary.json",
        {
            "impl": impl_label,
            "num_stages": D,
            "num_microbatches": N,
            "bubble_count": bubble,
            "bubble_ratio": bubble_ratio,
            "accept": accept_ok,
            "schedule_head": schedule[0][:20],
            "schedule_last": schedule[-1][:20],
        },
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
生成 D={D}, N={N} 的 1F1B schedule，并测算 bubble ratio。

## 2. 结果
- bubble count: {bubble}（理论 2(D-1)={2 * (D - 1)}）
- bubble ratio: {bubble_ratio:.4f}
- total ops: {total_ops}
- accept: {accept_ok}

## 3. 下一步
- 在 Megatron 上对照 `--pipeline-model-parallel-size {D}`
- N 越大 bubble_ratio 越小，但 microbatch 太大会损害 BS 收敛
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
