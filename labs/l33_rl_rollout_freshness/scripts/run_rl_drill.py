"""L11.5 · Drive student's versioned RolloutManager through a tiny RL loop."""

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

MISSION_ID = "l33_rl_rollout_freshness"


def _impl():
    try:
        from starter import rollout_manager as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import rollout_manager as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _drill(mod, server_ids, prompts, max_staleness, total_steps, update_rate, subset_size, seed):
    rng = random.Random(seed)
    servers = [mod.RolloutServer(sid, weight_version=0) for sid in server_ids]
    manager = mod.RolloutManager(servers, max_staleness=max_staleness)
    failures = 0
    staleness_samples: list[int] = []
    for step in range(1, total_steps + 1):
        if rng.random() < update_rate:
            updates = rng.sample(server_ids, k=min(subset_size, len(server_ids)))
            manager.update_weights(step, server_ids=updates)
        try:
            data = manager.generate(prompts, actor_version=step)
            staleness_samples.append(int(data.meta_info.get("staleness", 0)))
        except RuntimeError:
            failures += 1
    return {
        "failures": failures,
        "staleness_p50": statistics.median(staleness_samples) if staleness_samples else None,
        "staleness_p99": (
            sorted(staleness_samples)[int(len(staleness_samples) * 0.99)]
            if staleness_samples
            else None
        ),
        "successful_rollouts": len(staleness_samples),
    }


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

    if "slime_train_command" in config:
        write_text(
            run_dir / "artifacts" / "slime_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["slime_train_command"].split())
            + " &\n"
            + " ".join(config["slime_rollout_command"].split())
            + "\n",
        )

    server_ids = list(config["servers"])
    prompts = list(config["prompts"])
    max_staleness = int(config["max_staleness"])
    total_steps = int(config["total_steps"])
    seed = int(config.get("seed", 0))

    nominal = _drill(
        mod,
        server_ids,
        prompts,
        max_staleness,
        total_steps,
        float(config["update_rate"]),
        int(config["update_subset_size"]),
        seed,
    )
    no_update = _drill(
        mod, server_ids, prompts, max_staleness, total_steps, 0.0, 0, seed
    )

    for label, summary in (("nominal", nominal), ("no_update", no_update)):
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "rl_drill",
                "label": label,
                **summary,
            },
        )

    acceptance = config.get("acceptance", {})
    nominal_ok = nominal["failures"] == 0 if float(config["update_rate"]) >= float(
        acceptance.get("no_runtime_error_when_update_rate_geq", 1.0)
    ) else True
    no_update_ok = no_update["failures"] >= 1
    p99_ok = (
        nominal["staleness_p99"] is None
        or nominal["staleness_p99"] <= int(acceptance.get("p99_staleness_max", 999))
    )
    overall = nominal_ok and no_update_ok and p99_ok

    write_json(
        run_dir / "artifacts" / "rl_drill.json",
        {
            "impl": impl_label,
            "all_passed": overall,
            "nominal_ok": nominal_ok,
            "no_update_ok": no_update_ok,
            "p99_ok": p99_ok,
            "nominal": nominal,
            "no_update": no_update,
        },
    )

    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
驱动 patch 的 `RolloutManager` 跑一段假 RL，量化 stale rollout 的发生率与
weight sync 节奏的关系。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- servers: {server_ids}
- max_staleness: {max_staleness}
- total_steps: {total_steps}
- update_rate: {config['update_rate']}, subset_size: {config['update_subset_size']}

## 3. 预测
- 充分同步（update_rate ≥ {acceptance.get('no_runtime_error_when_update_rate_geq', 1.0)}）下不应抛错
- update_rate=0 时 manager 必须抛 RuntimeError，不能默默使用 stale 权重
- p99 staleness ≤ {acceptance.get('p99_staleness_max', 999)}

## 4. 运行命令
见 `command.sh`；profile 含 SLiME 启动模板时另见 `artifacts/slime_command.sh`。

## 5. 结果
- nominal: {nominal}
- no_update（必须有 failure）: {no_update}
- 全部通过： {overall}

## 6. 诊断
- 同步充分仍报错：检查 fresh-enough 计算是否包含 actor_version 边界
- update_rate=0 仍能 generate：patch 没有抛 RuntimeError，违反 stale-safe 合同
- p99 staleness 太大：subset_size 太小或 max_staleness 太宽

## 7. Debug 工单
推荐 `slime_stale_rollout_silent`、`slime_weight_sync_stall`、`slime_subset_starvation`。

## 8. PR Review
评审 patch 时确认：
- staleness = actor_version - server.weight_version
- 没有 fresh server 必须抛错
- update_weights(version, server_ids) 仅更新指定子集

## 9. 我原来误解了什么
（学习者自填）

## 10. 如果迁移到真实 SLiME
按 `artifacts/slime_command.sh` 启动 actor + rollout 集群，把 staleness 和
update 时间戳同时输出到 wandb。

## 11. 下一步
进入 L12 capstone。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
