"""L07.5 · Drive student's vLLM-shaped scheduler/KV manager through a synthetic workload."""

from __future__ import annotations

import argparse
import random
import statistics
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

MISSION_ID = "l20_vllm_scheduler_kv"


def _impl():
    try:
        from starter import scheduler as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import scheduler as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _build_workload(config: dict, mod) -> list:
    if "workload" in config:
        return [
            mod.Request(
                request_id=str(item["request_id"]),
                prompt=" ".join(["tok"] * int(item["prompt_tokens"])),
                max_tokens=int(item["max_tokens"]),
            )
            for item in config["workload"]
        ]
    rng = random.Random(int(config.get("workload_seed", 0)))
    size = int(config["workload_size"])
    prompt_lo, prompt_hi = config["workload_prompt_tokens_range"]
    out_lo, out_hi = config["workload_max_tokens_range"]
    requests = []
    for idx in range(size):
        prompt_tokens = rng.randint(int(prompt_lo), int(prompt_hi))
        max_tokens = rng.randint(int(out_lo), int(out_hi))
        requests.append(
            mod.Request(
                request_id=f"r{idx}",
                prompt=" ".join(["tok"] * prompt_tokens),
                max_tokens=max_tokens,
            )
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=2000)
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

    if "vllm_serve_command" in config:
        write_text(
            run_dir / "artifacts" / "vllm_serve_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["vllm_serve_command"].split())
            + "\n",
        )

    kv = mod.KVCacheManager(num_blocks=int(config["scheduler"]["num_blocks"]))
    scheduler = mod.Scheduler(kv, max_num_running_reqs=int(config["scheduler"]["max_num_running_reqs"]))
    requests = _build_workload(config, mod)
    pending = list(requests)
    arrival_step: dict[str, int] = {}
    first_decode_step: dict[str, int] = {}
    decode_steps_per_request: dict[str, int] = {}
    finish_step: dict[str, int] = {}
    waiting_lengths: list[int] = []
    kv_usages: list[float] = []
    waiting_grew = False

    step = 0
    while step < args.max_steps:
        step += 1
        # admit a few new requests per step (Poisson-ish)
        admit_count = max(1, len(requests) // 16)
        for _ in range(admit_count):
            if not pending:
                break
            req = pending.pop(0)
            scheduler.add_request(req)
            arrival_step[req.request_id] = step

        out = scheduler.schedule()

        # produce one token per running request (decode tick)
        for req_id in out.scheduled_decode:
            req = scheduler.running.get(req_id)
            if req is not None:
                req.output_tokens.append("tok")
                first_decode_step.setdefault(req_id, step)
                decode_steps_per_request[req_id] = decode_steps_per_request.get(req_id, 0) + 1

        for req_id in out.finished:
            finish_step[req_id] = step

        waiting_lengths.append(len(scheduler.waiting))
        kv_usages.append(float(kv.usage))
        if len(scheduler.waiting) > 0 and float(kv.usage) > 0.7:
            waiting_grew = True

        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "schedule_tick",
                "step": step,
                "kv_usage": float(kv.usage),
                "running": len(scheduler.running),
                "waiting": len(scheduler.waiting),
                "scheduled_prefill": out.scheduled_prefill,
                "scheduled_decode_count": len(out.scheduled_decode),
                "finished": out.finished,
            },
        )
        if not pending and not scheduler.waiting and not scheduler.running:
            break

    ttft = sorted(
        first_decode_step[req_id] - arrival_step[req_id]
        for req_id in first_decode_step
        if req_id in arrival_step
    )
    itl = sorted(
        (finish_step[req_id] - first_decode_step[req_id]) / max(1, decode_steps_per_request[req_id])
        for req_id in finish_step
        if req_id in first_decode_step
    )

    p50_kv = statistics.median(kv_usages) if kv_usages else 0.0
    p99_kv = max(kv_usages) if kv_usages else 0.0
    all_terminated = len(finish_step) == len(requests)
    acceptance = config.get("acceptance", {})
    min_p50 = float(acceptance.get("min_p50_kv_usage", 0.0))
    require_waiting = bool(acceptance.get("waiting_must_grow_under_pressure", False))
    must_terminate = bool(acceptance.get("must_terminate", True))

    accept_p50 = p50_kv >= min_p50
    accept_term = (not must_terminate) or all_terminated
    accept_waiting = (not require_waiting) or waiting_grew or p99_kv < 0.9
    overall = accept_p50 and accept_term and accept_waiting

    write_json(
        run_dir / "artifacts" / "ttft_itl.json",
        {
            "p50_ttft": ttft[len(ttft) // 2] if ttft else None,
            "p95_ttft": ttft[int(len(ttft) * 0.95)] if ttft else None,
            "p50_itl": itl[len(itl) // 2] if itl else None,
            "p50_kv_usage": p50_kv,
            "p99_kv_usage": p99_kv,
            "requests_total": len(requests),
            "requests_finished": len(finish_step),
            "waiting_grew_under_pressure": waiting_grew,
            "acceptance": {
                "p50_kv_ok": accept_p50,
                "terminated_ok": accept_term,
                "waiting_pressure_ok": accept_waiting,
                "all_passed": overall,
            },
        },
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
驱动 patch 的 `Scheduler` + `KVCacheManager` 完整跑一个 serving workload。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- num_blocks: {config['scheduler']['num_blocks']}
- max_num_running_reqs: {config['scheduler']['max_num_running_reqs']}
- requests: {len(requests)}

## 3. 预测
- p50 KV usage 应 >= {min_p50}
- 所有 request 应在 {step} step 内完成
- 高负载下 waiting 队列必须有时增长（不能默默丢请求）

## 4. 运行命令
见 `command.sh`；若 profile 含 `vllm_serve_command`，真实启动命令在
`artifacts/vllm_serve_command.sh`。

## 5. 结果
- requests finished: {len(finish_step)} / {len(requests)}
- p50 TTFT (steps): {ttft[len(ttft) // 2] if ttft else None}
- p95 TTFT (steps): {ttft[int(len(ttft) * 0.95)] if ttft else None}
- p50 ITL (steps/token): {itl[len(itl) // 2] if itl else None}
- p50 KV usage: {p50_kv:.3f}
- p99 KV usage: {p99_kv:.3f}
- waiting grew under pressure: {waiting_grew}
- acceptance all passed: {overall}

## 6. 诊断
- TTFT 偏大：检查 admit 路径里是不是被 KV 卡住，看 `kv_cache_manager.usage`。
- ITL 偏大：检查 `scheduled_decode` 是不是每 step 都包含 running request。
- 高 KV usage 下 waiting=0：scheduler 可能默默丢请求或乱共享 KV blocks。

## 7. Debug 工单
推荐 `vllm_kv_oom_under_burst`、`vllm_waiting_starvation`、`vllm_decode_drop`。

## 8. PR Review
评审 patch 时确认：
- `add_request` 不直接占 KV
- prefill 阶段才 `allocate_slots`
- `finished` 必须 `free` KV blocks
- `max_num_running_reqs` 与 KV 容量需要双重门禁

## 9. 我原来误解了什么
（学习者自填）

## 10. 如果迁移到真实 vLLM
按 `artifacts/vllm_serve_command.sh` 启动 OpenAI 兼容服务，再用
`scripts/bench_ttft.py --base-url http://localhost:8000` 做真实 TTFT/ITL 测试。

## 11. 下一步
进入 L08 学 SGLang RadixCache + L08.7 spec decode。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
