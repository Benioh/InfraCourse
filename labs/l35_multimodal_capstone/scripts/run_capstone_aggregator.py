from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

MISSION_ID = "l35_multimodal_capstone"
LAB_DIR = Path(__file__).resolve().parents[1]
QUESTS_DIR = ROOT / "quests"
RUNS_DIR = ROOT / "runs"
TICKETS_DIR = ROOT / "tickets"

STAGE_RULES = {
    "env": ["l00_", "l01_", "l02_"],
    "training": ["l03_", "l04_", "l05_", "l06_"],
    "serving": ["l07_", "l08_", "l09_"],
    "rl": ["l10_", "l11_"],
}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": line[:200]})
    return rows


def list_quests() -> list[dict[str, Any]]:
    quests = [read_yaml(path) for path in sorted(QUESTS_DIR.glob("*.yaml"))]
    return sorted(
        quests, key=lambda item: (str(item.get("level", "")), str(item.get("id", "")))
    )


def latest_run(mission_id: str) -> Path | None:
    run_root = RUNS_DIR / mission_id
    if not run_root.exists():
        return None
    runs = sorted(path for path in run_root.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def command_body(command_path: Path) -> str:
    if not command_path.exists():
        return ""
    lines = []
    for line in command_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("set -")
            or stripped.startswith("#!/")
        ):
            continue
        lines.append(stripped)
    return " && ".join(lines)


def stage_for(mission_id: str) -> str:
    for stage, prefixes in STAGE_RULES.items():
        if any(mission_id.startswith(prefix) for prefix in prefixes):
            return stage
    return "other"


def collect_ticket_ids() -> set[str]:
    return {path.stem for path in TICKETS_DIR.glob("*.yaml")}


def mentioned_tickets(text: str, ticket_ids: set[str]) -> list[str]:
    return sorted(ticket for ticket in ticket_ids if ticket in text)


def summarize_mission(quest: dict[str, Any], ticket_ids: set[str]) -> dict[str, Any]:
    mission_id = quest["id"]
    if mission_id == MISSION_ID:
        return {"mission": mission_id, "skip": True}
    run = latest_run(mission_id)
    if run is None:
        return {
            "mission": mission_id,
            "level": quest.get("level"),
            "title": quest.get("title"),
            "stage": stage_for(mission_id),
            "run_id": None,
            "run_path": None,
            "has_run": False,
            "missing_reason": "no run directory found",
        }

    metrics = read_jsonl(run / "metrics.jsonl")
    grade = read_json(run / "grade.json")
    config = (
        read_yaml(run / "config.resolved.yaml")
        if (run / "config.resolved.yaml").exists()
        else {}
    )
    report_text = (
        (run / "report.md").read_text(encoding="utf-8")
        if (run / "report.md").exists()
        else ""
    )
    artifacts = (
        sorted(
            str(path.relative_to(run))
            for path in (run / "artifacts").rglob("*")
            if path.is_file()
        )
        if (run / "artifacts").exists()
        else []
    )
    return {
        "mission": mission_id,
        "level": quest.get("level"),
        "title": quest.get("title"),
        "stage": stage_for(mission_id),
        "run_id": run.name,
        "run_path": str(run.relative_to(ROOT)),
        "has_run": True,
        "mode": config.get("mode"),
        "command": command_body(run / "command.sh"),
        "config_path": (
            str((run / "config.resolved.yaml").relative_to(ROOT))
            if (run / "config.resolved.yaml").exists()
            else None
        ),
        "metrics_path": (
            str((run / "metrics.jsonl").relative_to(ROOT))
            if (run / "metrics.jsonl").exists()
            else None
        ),
        "report_path": (
            str((run / "report.md").relative_to(ROOT))
            if (run / "report.md").exists()
            else None
        ),
        "grade_passed": bool(grade and grade.get("passed")),
        "grade_score": grade.get("score") if grade else None,
        "latest_metric": metrics[-1] if metrics else None,
        "artifact_count": len(artifacts),
        "artifacts_sample": artifacts[:8],
        "tickets_mentioned": mentioned_tickets(report_text, ticket_ids),
        "validation_only": "validation-only" in report_text.lower()
        or "validation_only" in report_text.lower(),
    }


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            if value is None:
                value = ""
            values.append(str(value).replace("|", "/"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_final_readme(
    final_dir: Path, evidence_rows: list[dict[str, Any]], stage_counts: dict[str, int]
) -> None:
    rows = [row for row in evidence_rows if not row.get("skip")]
    content = f"""# Final Infra Delivery README

本文件由 L12 Capstone 聚合器生成，来源是 `runs/` 中 L00-L11 的最新 run。它不是空模板：每一行都应能回到 command/config/metrics/report 证据。

## 汇总

- 生成时间：{utc_now()}
- 前序任务数：{len(rows)}
- 已发现 run：{sum(1 for row in rows if row.get('has_run'))}
- 阶段覆盖：{stage_counts}

## Evidence Index

{table(rows, [('Level', 'level'), ('Mission', 'mission'), ('Stage', 'stage'), ('Run', 'run_id'), ('Mode', 'mode'), ('Grade', 'grade_score'), ('Validation-only', 'validation_only'), ('Report', 'report_path')])}

## 迁移判断

- Bronze：有 run、command/config/metrics/report 证据，但可能是 validation-only。
- Silver：同一阶段至少有 baseline + 对照 run，并完成源码改造任务。
- Gold：真实框架或 8×H200 运行结果能支撑性能/稳定性结论。

## 下一步

1. 对缺 run 的 mission 先执行对应 `make smoke M=<mission>`。
2. 对 validation-only 行补真实 server/train/RL 运行，或在风险登记中保留边界。
3. 把 L04/L07/L10/L11 的源码改造任务产物加入对应 run artifacts。
"""
    write_text(final_dir / "final_readme.md", content)


def write_repro_script(final_dir: Path, evidence_rows: list[dict[str, Any]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by L12 Capstone aggregator",
        "make check-env",
        "",
    ]
    for row in evidence_rows:
        if row.get("skip"):
            continue
        mission = row["mission"]
        if row.get("command"):
            lines.append(f"# source run: {row.get('run_path')}")
            lines.append(f"# original command: {row['command']}")
        else:
            lines.append(f"# no previous run found for {mission}")
        lines.append(f"make smoke M={mission}")
        lines.append("")
    lines.append("make smoke M=l35_multimodal_capstone")
    write_text(final_dir / "reproducibility_commands.sh", "\n".join(lines) + "\n")
    (final_dir / "reproducibility_commands.sh").chmod(0o755)


def write_debug_report(final_dir: Path, evidence_rows: list[dict[str, Any]]) -> int:
    rows = []
    unique_tickets: set[str] = set()
    for row in evidence_rows:
        for ticket in row.get("tickets_mentioned", []):
            unique_tickets.add(ticket)
            rows.append(
                {
                    "ticket": ticket,
                    "mission": row["mission"],
                    "run_id": row.get("run_id"),
                    "report_path": row.get("report_path"),
                }
            )
    if not rows:
        rows.append(
            {
                "ticket": "",
                "mission": "",
                "run_id": "",
                "report_path": "未在前序报告中发现 ticket；请补 debug 证据。",
            }
        )
    content = f"""# Debug Report

本报告由前序 `report.md` 中出现的 ticket ID 聚合生成。若为空，说明 Capstone 还不能证明 debug 覆盖度。

{table(rows, [('Ticket', 'ticket'), ('Mission', 'mission'), ('Run', 'run_id'), ('Report', 'report_path')])}

## 要求

至少覆盖环境、训练、数据、Serving、RL 五类 failure mode；每个 ticket 都要能回到最小检查、修复动作和 evidence path。
"""
    write_text(final_dir / "debug_report.md", content)
    return len(unique_tickets)


def write_stage_reports(
    final_dir: Path, evidence_rows: list[dict[str, Any]]
) -> dict[str, int]:
    stage_dir = final_dir / "stage_reports"
    stage_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for stage in ["env", "training", "serving", "rl", "other"]:
        rows = [
            row
            for row in evidence_rows
            if row.get("stage") == stage and not row.get("skip")
        ]
        counts[stage] = sum(1 for row in rows if row.get("has_run"))
        content = f"""# {stage.title()} Stage Report

## Evidence

{table(rows, [('Level', 'level'), ('Mission', 'mission'), ('Run', 'run_id'), ('Mode', 'mode'), ('Metrics', 'metrics_path'), ('Report', 'report_path')])}

## Gap Review

- 缺 run 的任务需要先补 smoke 或真实运行。
- validation-only 的任务不能支撑真实性能结论。
- Silver/Gold 需要补对照实验和源码改造任务证据。
"""
        write_text(stage_dir / f"{stage}.md", content)
    return counts


def write_risk_register(final_dir: Path, evidence_rows: list[dict[str, Any]]) -> None:
    missing = [
        row for row in evidence_rows if not row.get("has_run") and not row.get("skip")
    ]
    validation = [row for row in evidence_rows if row.get("validation_only")]
    risks = [
        {
            "risk": "前序 run 缺失",
            "count": len(missing),
            "mitigation": "按 reproducibility_commands.sh 补 smoke/真实运行",
        },
        {
            "risk": "validation-only 结论",
            "count": len(validation),
            "mitigation": "补 4090/H200 真实运行或保留边界",
        },
        {
            "risk": "源码改造任务缺证据",
            "count": 4,
            "mitigation": "检查 L04/L07/L10/L11 的 source_patch_task.md 产物",
        },
    ]
    content = f"""# Risk Register

{table(risks, [('Risk', 'risk'), ('Count', 'count'), ('Mitigation', 'mitigation')])}

## 使用要求

每个风险必须连接到具体 mission、run_id 和 evidence path；不能用 Capstone 模板替代前序证据。
"""
    write_text(final_dir / "risk_register.md", content)


def write_architecture(final_dir: Path) -> None:
    write_text(
        final_dir / "architecture.mmd",
        """flowchart LR
  Env[L00-L02 environment/distributed evidence] --> Train[L03-L06 training/data evidence]
  Train --> Ckpt[checkpoint + tokenizer/processor boundary]
  Ckpt --> Serving[L07-L09 serving evidence]
  Ckpt --> RL[L10-L11 RL evidence]
  Serving --> Metrics[metrics/logs/traces]
  RL --> Metrics
  Metrics --> Capstone[L12 evidence_index + risk_register]
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate L00-L11 runs into L12 final artifacts"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="capstone")
    args = parser.parse_args()

    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, "aggregation_source": "runs/*/*"},
    )

    ticket_ids = collect_ticket_ids()
    evidence_rows = [summarize_mission(quest, ticket_ids) for quest in list_quests()]
    evidence_rows = [row for row in evidence_rows if not row.get("skip")]
    final_dir = run_dir / "artifacts" / "final_artifacts"
    final_dir.mkdir(parents=True, exist_ok=True)

    stage_counts = write_stage_reports(final_dir, evidence_rows)
    write_final_readme(final_dir, evidence_rows, stage_counts)
    write_repro_script(final_dir, evidence_rows)
    tickets_completed = write_debug_report(final_dir, evidence_rows)
    write_risk_register(final_dir, evidence_rows)
    write_architecture(final_dir)
    write_json(final_dir / "evidence_index.json", evidence_rows)

    source_run_count = sum(1 for row in evidence_rows if row.get("has_run"))
    artifact_count = len([path for path in final_dir.rglob("*") if path.is_file()])
    pipeline_stages_ready = sum(
        1
        for stage in ["env", "training", "serving", "rl"]
        if stage_counts.get(stage, 0) > 0
    )
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "capstone",
            "artifact_count": artifact_count,
            "source_run_count": source_run_count,
            "tickets_completed": tickets_completed,
            "pipeline_stages_ready": pipeline_stages_ready,
            "reproducibility_ready": source_run_count > 0,
            "evidence_index_ready": True,
        },
    )
    write_text(
        run_dir / "train.log",
        f"[{utc_now()}] aggregated {source_run_count} source runs into {final_dir}\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
聚合 L00-L11 的真实 run 证据，形成 Capstone 可交付包，而不是生成空模板。

## 2. 源码调用链
`make smoke` → `scripts/run_capstone_aggregator.py` → 扫描 `runs/*/*` → 读取 command/config/metrics/report/grade → 写入 `artifacts/final_artifacts/evidence_index.json`、`final_readme.md` 和 stage reports。

## 3. 实验矩阵
| run_id | 只改变的变量 | 关键指标 | 结论 |
|---|---|---|---|
| {run_dir.name} | aggregation_source=runs | source_run_count={source_run_count} | Capstone 聚合了前序 run 证据 |
| {run_dir.name} | stage reports | pipeline_stages_ready={pipeline_stages_ready} | 阶段覆盖见 final_artifacts/stage_reports |

## 4. 结果
- artifact_count：{artifact_count}
- source_run_count：{source_run_count}
- tickets_completed：{tickets_completed}
- pipeline_stages_ready：{pipeline_stages_ready}
- evidence path：`artifacts/final_artifacts/evidence_index.json`

## 5. Debug Ticket
聚合器从前序报告中抽取 ticket ID；缺失时必须回到对应 lab 补最小检查和修复证据。Capstone 默认风险覆盖从 `mgt_oom_001`、`sglang_high_ttft_001`、`verl_reward_parse_001`、`mm_bad_shard_003`、`slime_rollout_bottleneck_001` 开始补齐。

## 6. 迁移判断
本次 Capstone 只聚合已有证据。若某行是 validation-only 或缺 run，不能支撑真实 8×H200、Serving 或 RL 成功；需要按 `reproducibility_commands.sh` 补跑。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
