from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def repo_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__).resolve()).resolve()
    for candidate in [probe, *probe.parents]:
        if (candidate / "INFRA_QUEST_BUILD_SPEC.md").exists():
            return candidate
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "README.md").exists()
            and (candidate / "labs").is_dir()
            and (candidate / "quests").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Could not locate repo root from INFRA_QUEST_BUILD_SPEC.md")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_run_dir(
    mission_id: str, run_id: str | None = None, root: Path | None = None
) -> Path:
    base = repo_root(root or Path(__file__).resolve()) / "runs" / mission_id
    ensure_dir(base)
    run_dir = base / (run_id or utc_run_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    return run_dir


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any, pretty: bool = True) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_run_dir(mission_id: str, run_id: str | None = None) -> Path:
    root = repo_root()
    mission_runs = root / "runs" / mission_id
    ensure_dir(mission_runs)
    if run_id:
        return mission_runs / run_id
    runs = [path for path in mission_runs.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found for mission {mission_id}")
    return sorted(runs)[-1]


def write_command_snapshot(
    run_dir: Path, argv: list[str] | None = None, env: dict[str, str] | None = None
) -> None:
    command = " ".join(argv or os.sys.argv)
    env_payload = env or {}
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {utc_now()}", command]
    if env_payload:
        lines.extend(["", "# environment"])
        lines.extend(
            f"export {key}={value}" for key, value in sorted(env_payload.items())
        )
    write_text(run_dir / "command.sh", "\n".join(lines) + "\n")


def default_prediction() -> dict[str, Any]:
    return {
        "expected_bottleneck": "unknown",
        "expected_peak_memory_gb": None,
        "expected_tokens_per_sec": None,
        "expected_failure_mode": "unknown",
        "confidence": 0.5,
    }


def ensure_prediction(path: Path) -> None:
    if not path.exists():
        write_yaml(path, default_prediction())
