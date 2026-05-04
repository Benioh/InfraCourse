"""L05.8 · Run save/load/mismatch drill against the student's checkpoint patch."""

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

MISSION_ID = "l15_megatron_parallel_checkpoint"


def _impl():
    try:
        from starter import checkpointing  # type: ignore[import-not-found]

        return checkpointing, "starter"
    except (ImportError, NotImplementedError):
        from reference import checkpointing  # type: ignore[import-not-found]

        return checkpointing, "reference"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    impl, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )

    ckpt_dir = run_dir / "artifacts" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_result = impl.save_checkpoint(
        ckpt_dir,
        iteration=int(config["checkpoint_iteration"]),
        model_state=config["states"]["model_state"],
        optimizer_state=config["states"]["optimizer_state"],
        scheduler_state=config["states"]["scheduler_state"],
        parallel_state=config["states"]["parallel_state"],
    )
    write_json(run_dir / "artifacts" / "save_result.json", save_result)

    drill_results: list[dict] = []
    overall_pass = True
    for case in config["expected_compatibility"]:
        label = case["label"]
        expected = case["parallel_state"]
        outcome: dict = {
            "label": label,
            "expected_parallel_state": expected,
            "must_pass_strict": case.get("must_pass_strict", True),
            "must_have_warnings": case.get("must_have_warnings"),
        }
        # strict
        try:
            payload = impl.load_checkpoint(ckpt_dir, expected_parallel_state=expected, strict=True)
            outcome["strict_raised"] = False
            outcome["strict_warnings"] = payload.get("warnings", [])
        except Exception as exc:  # noqa: BLE001
            outcome["strict_raised"] = True
            outcome["strict_error"] = type(exc).__name__
        # non-strict
        payload = impl.load_checkpoint(ckpt_dir, expected_parallel_state=expected, strict=False)
        outcome["non_strict_warnings"] = payload.get("warnings", [])

        case_ok = True
        if case.get("must_pass_strict") is True and outcome["strict_raised"]:
            case_ok = False
        if case.get("must_pass_strict") is False and not outcome["strict_raised"]:
            case_ok = False
        if (
            case.get("must_have_warnings") is not None
            and len(outcome["non_strict_warnings"]) < int(case["must_have_warnings"])
        ):
            case_ok = False
        outcome["case_ok"] = case_ok
        overall_pass = overall_pass and case_ok
        drill_results.append(outcome)

        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "checkpoint_drill",
                "case": label,
                "case_ok": case_ok,
                "strict_raised": outcome["strict_raised"],
                "non_strict_warnings": len(outcome["non_strict_warnings"]),
            },
        )

    write_json(
        run_dir / "artifacts" / "drill.json",
        {"impl": impl_label, "all_passed": overall_pass, "cases": drill_results},
    )

    save_cmd = config.get("megatron_save_command")
    load_cmd = config.get("megatron_load_command")
    if save_cmd or load_cmd:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        if save_cmd:
            lines.append("# save")
            lines.append(" ".join(save_cmd.split()))
        if load_cmd:
            lines.append("# load")
            lines.append(" ".join(load_cmd.split()))
        write_text(run_dir / "artifacts" / "real_megatron_cmd.sh", "\n".join(lines) + "\n")

    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
演练 distributed checkpoint 的 save/load/parallel_state 校验。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- iteration: {config['checkpoint_iteration']}
- saved parallel_state: {config['states']['parallel_state']}

## 3. 预测
- 同拓扑 strict load 必须无 warning
- 改变 TP 必须 strict 抛错，non-strict 至少 1 条 warning

## 4. 运行命令
见 `command.sh`；真实 Megatron save/load 命令在 `artifacts/real_megatron_cmd.sh`。

## 5. 结果
- 全部 case 通过： {overall_pass}
- 详见 `artifacts/drill.json`

## 6. 诊断
若 strict=True 不抛错，说明 patch 漏掉 parallel_state 比对；
若 non-strict 没有 warning，说明 patch 没有把不匹配项收集到 warnings；
若 latest marker 缺失，回去检查 save_checkpoint 是否写了 `latest_checkpointed_iteration.txt`。

## 7. Debug 工单
推荐 `ckpt_tp_mismatch`、`ckpt_resume_lr_jump`、`ckpt_missing_marker`。

## 8. PR Review
评审 patch 时确认：
- save 必须写 latest marker
- load 必须根据 latest marker 找文件
- strict / non-strict 行为对称
- 比较项不要遗漏 ep / sequence_parallel 等新维度

## 9. 我原来误解了什么
（学习者自填）

## 10. 如果迁移到 8×H200
执行 `artifacts/real_megatron_cmd.sh`，把 metadata 校验逻辑接到 Megatron 自己的
`get_checkpoint_name` / `read_metadata` 流程。

## 11. 下一步
进入 L06 学多模态数据组装。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
