"""L09.5 · Drive student's DisaggregationService through a synthetic workload."""

from __future__ import annotations

import argparse
import random
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

MISSION_ID = "l27_sglang_pd_cache"


def _impl():
    try:
        from starter import disagg_service as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import disagg_service as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _generate_workload(config: dict) -> list[dict]:
    rng = random.Random(int(config.get("seed", 0)))
    requests = []
    lo, hi = config["prompt_tokens_range"]
    cache_rate = float(config["cache_hit_rate"])
    hit_ratio = float(config["cached_prefix_ratio_when_hit"])
    for idx in range(int(config["num_requests"])):
        prompt = rng.randint(int(lo), int(hi))
        cached = int(prompt * hit_ratio) if rng.random() < cache_rate else 0
        requests.append({"id": f"req-{idx}", "prompt": prompt, "cached": cached})
    return requests


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

    if "sglang_prefill_command" in config:
        cmd = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "echo '=== prefill ==='",
                " ".join(config["sglang_prefill_command"].split()) + " &",
                "echo '=== decode ==='",
                " ".join(config["sglang_decode_command"].split()) + " &",
                "echo '=== router ==='",
                " ".join(config["sglang_router_command"].split()),
            ]
        )
        write_text(run_dir / "artifacts" / "sglang_pd_command.sh", cmd + "\n")

    workload_off = _generate_workload(
        {**config["workload"], "cache_hit_rate": 0.0}
    )
    workload_on = _generate_workload(config["workload"])

    def run_pass(label: str, workload: list[dict]) -> dict:
        service = mod.DisaggregationService(
            prefill_workers=list(config["service"]["prefill_workers"]),
            decode_workers=list(config["service"]["decode_workers"]),
        )
        for req in workload:
            route = service.route_request(
                req["id"], int(req["prompt"]), cached_prefix_tokens=int(req["cached"])
            )
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "timestamp": utc_now(),
                    "metric_type": "route",
                    "pass": label,
                    "request_id": req["id"],
                    "prompt_tokens": int(req["prompt"]),
                    "cached_prefix_tokens": int(req["cached"]),
                    "prefill_tokens": route.prefill_tokens,
                    "prefill_worker": route.prefill_worker,
                    "decode_worker": route.decode_worker,
                },
            )
        snap_before_complete = service.metrics()
        for req in workload:
            service.complete_request(req["id"])
        return {
            "label": label,
            "snapshot": snap_before_complete,
            "drained": service.metrics(),
            "transfers_per_request_min": min(
                len(service.transfers_for_request(req["id"])) for req in workload
            ),
            "transfers_per_request_max": max(
                len(service.transfers_for_request(req["id"])) for req in workload
            ),
        }

    summary_off = run_pass("cache_off", workload_off)
    summary_on = run_pass("cache_on", workload_on)

    saving = (
        (summary_off["snapshot"]["prefill_tokens"] - summary_on["snapshot"]["prefill_tokens"])
        / max(1, summary_off["snapshot"]["prefill_tokens"])
    )

    acceptance = config.get("acceptance", {})
    saving_ok = saving >= float(acceptance.get("min_prefill_saving_ratio", 0.0))
    one_transfer_ok = (
        summary_on["transfers_per_request_min"] == 1
        and summary_on["transfers_per_request_max"] == 1
    )
    drained_loads = summary_on["drained"]["worker_loads"].values()
    drained_ok = all(value == 0 for value in drained_loads)

    overall = saving_ok and one_transfer_ok and (
        not acceptance.get("workers_must_drain_on_complete", True) or drained_ok
    )

    write_json(
        run_dir / "artifacts" / "pd_drill.json",
        {
            "impl": impl_label,
            "saving_ratio": saving,
            "saving_ok": saving_ok,
            "one_transfer_ok": one_transfer_ok,
            "drained_ok": drained_ok,
            "all_passed": overall,
            "cache_off": summary_off,
            "cache_on": summary_on,
        },
    )

    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
驱动 patch 的 `DisaggregationService` 完整跑一遍 prefill/decode 分离的 PD workload。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- prefill workers: {config['service']['prefill_workers']}
- decode workers: {config['service']['decode_workers']}
- request 数: {config['workload']['num_requests']}

## 3. 预测
- prefix cache on 时 prefill_tokens 应减少 ≥ {acceptance.get('min_prefill_saving_ratio', 0.0)}
- 每个 request 应恰好 1 条 KV transfer
- complete 后 worker_loads 应归零

## 4. 运行命令
见 `command.sh`；profile 含 SGLang 启动模板时另见 `artifacts/sglang_pd_command.sh`。

## 5. 结果
- prefill saving (cache off→on): {saving:.3f}（要求 ≥ {acceptance.get('min_prefill_saving_ratio', 0.0)}）
- transfers/request（on）: [{summary_on['transfers_per_request_min']}, {summary_on['transfers_per_request_max']}]
- drained worker_loads OK: {drained_ok}
- 全部通过： {overall}

## 6. 诊断
- saving 不达标：检查 `prefill_tokens = prompt - cached_prefix_tokens` 公式
- transfers > 1：route 路径里有重复 transfer_kv 调用
- workers 不归零：complete_request 漏掉了某 worker 计数

## 7. Debug 工单
推荐 `pd_router_misroute`、`pd_kv_transfer_lost`、`pd_complete_double`。

## 8. PR Review
评审 patch 时确认：
- route + transfer_kv 一一对应
- complete_request 必须幂等
- metrics 同时暴露 prefill / cached / transferred 三层指标

## 9. 我原来误解了什么
（学习者自填）

## 10. 如果迁移到真实 SGLang
按 `artifacts/sglang_pd_command.sh` 启动 prefill / decode / router 三个进程，再用
`bench_ttft.py` 打流量，关键指标和本 drill 同名。

## 11. 下一步
进入 L10 学 verl RL baseline。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
