from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from app.backend.content import CONCEPT_MAP, SOURCE_MAPS

ROOT = Path(__file__).resolve().parents[2]
QUESTS_DIR = ROOT / "quests"
TICKETS_DIR = ROOT / "tickets"
PROMPTS_DIR = ROOT / "prompts"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def quest_sort_key(quest: dict[str, Any]) -> tuple[str, str]:
    return (quest.get("level", ""), quest.get("id", ""))


def list_quests() -> list[dict[str, Any]]:
    quests = [read_yaml(path) for path in sorted(QUESTS_DIR.glob("*.yaml"))]
    return sorted(quests, key=quest_sort_key)


def get_quest(mission_id: str) -> dict[str, Any]:
    path = QUESTS_DIR / f"{mission_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(mission_id)
    return read_yaml(path)


def list_tickets() -> list[dict[str, Any]]:
    tickets: list[dict[str, Any]] = []
    for path in sorted(TICKETS_DIR.glob("*.yaml")):
        try:
            data = read_yaml(path)
        except yaml.YAMLError as exc:
            tickets.append({
                "id": path.stem,
                "title": f"[YAML parse error] {path.name}",
                "severity": "unknown",
                "_parse_error": str(exc).splitlines()[0],
            })
            continue
        if isinstance(data, dict):
            tickets.append(data)
    return sorted(tickets, key=lambda item: item.get("id", ""))


def get_ticket(ticket_id: str) -> dict[str, Any]:
    path = TICKETS_DIR / f"{ticket_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(ticket_id)
    return read_yaml(path)


# ---------------------------------------------------------------------------
# ★ Quiz Gate helpers — gate before students can write the patch.
# ---------------------------------------------------------------------------


LABS_DIR = ROOT / "labs"


def _quiz_yaml_path(mission_id: str) -> Path:
    return LABS_DIR / mission_id / "quiz.yaml"


def _quiz_status_path(mission_id: str) -> Path:
    return LABS_DIR / mission_id / "patch" / ".quiz_passed.json"


def _read_quiz_yaml(mission_id: str) -> dict[str, Any] | None:
    path = _quiz_yaml_path(mission_id)
    if not path.exists():
        return None
    return read_yaml(path)


def quiz_payload(mission_id: str) -> dict[str, Any]:
    """Return the quiz **without `correct` markers**, plus prereq checklist
    and the current pass status. Frontend renders this for the student."""
    quiz = _read_quiz_yaml(mission_id)
    if not quiz:
        raise FileNotFoundError(f"no quiz.yaml for {mission_id}")

    safe_questions = []
    for q in quiz.get("questions", []):
        safe_questions.append(
            {
                "id": q["id"],
                "prompt": q.get("prompt", ""),
                "options": [
                    {"id": opt["id"], "text": opt.get("text", "")}
                    for opt in q.get("options", [])
                ],
            }
        )

    status_path = _quiz_status_path(mission_id)
    status = _read_patch_status_file(status_path) if status_path.exists() else None

    return {
        "mission": mission_id,
        "title": quiz.get("title"),
        "description": quiz.get("description"),
        "pass_threshold": quiz.get("pass_threshold", 1.0),
        "prereq_checklist": quiz.get("prereq_checklist", []),
        "questions": safe_questions,
        "total_questions": len(safe_questions),
        "status": status,  # null if never passed
    }


def submit_quiz(mission_id: str, answers: dict[str, str]) -> dict[str, Any]:
    """Grade the submitted answers, persist pass status if 100% (or threshold).

    `answers` is `{question_id: chosen_option_id}`.
    Returns per-question results with explanations (always shown, regardless of pass)."""
    quiz = _read_quiz_yaml(mission_id)
    if not quiz:
        raise FileNotFoundError(f"no quiz.yaml for {mission_id}")

    results = []
    correct_count = 0
    for q in quiz.get("questions", []):
        qid = q["id"]
        expected = next(
            (opt["id"] for opt in q.get("options", []) if opt.get("correct")), None
        )
        chosen = answers.get(qid)
        is_correct = chosen is not None and chosen == expected
        if is_correct:
            correct_count += 1
        results.append(
            {
                "id": qid,
                "your_answer": chosen,
                "expected": expected,
                "correct": is_correct,
                "explanation": q.get("explanation", ""),
            }
        )

    total = len(results)
    score = correct_count / total if total else 0.0
    threshold = float(quiz.get("pass_threshold", 1.0))
    passed = score >= threshold

    payload = {
        "mission": mission_id,
        "score": correct_count,
        "total": total,
        "score_ratio": round(score, 4),
        "threshold": threshold,
        "passed": passed,
        "results": results,
        "submitted_at": _utc_now_iso(),
    }

    if passed:
        # Persist pass status. Only writes on success — wrong answers don't get logged.
        status_path = _quiz_status_path(mission_id)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "mission": mission_id,
                    "passed": True,
                    "score": correct_count,
                    "total": total,
                    "passed_at": payload["submitted_at"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return payload


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")


def quiz_status(mission_id: str) -> dict[str, Any]:
    path = _quiz_status_path(mission_id)
    if not path.exists():
        return {"mission": mission_id, "passed": False, "never_attempted": True}
    return read_json(path)


# ---------------------------------------------------------------------------
# ★ Patch Track helpers — single source of truth for "did this lab pass?"
# ---------------------------------------------------------------------------


def _resolve_patch_status_path(quest: dict[str, Any]) -> Path | None:
    patch = quest.get("patch") or {}
    rel = patch.get("status_file")
    if not rel:
        return None
    return ROOT / rel


def _read_patch_status_file(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:  # noqa: BLE001 — corrupt status file shouldn't crash UI
        return None


def patch_status(mission_id: str) -> dict[str, Any]:
    """Read the .last_run.json written by scripts/run_patch_test.py.

    Raises FileNotFoundError when the lab has never had a patch test run."""
    quest = get_quest(mission_id)
    status_path = _resolve_patch_status_path(quest)
    status = _read_patch_status_file(status_path)
    if not status:
        raise FileNotFoundError(f"no patch status for {mission_id}")
    return status


# ---------------------------------------------------------------------------
# Starter file helpers — read/write the editable starter for a mission.
# ---------------------------------------------------------------------------


STARTER_BYTE_LIMIT = 200_000


def _starter_path(mission_id: str) -> Path:
    quest = get_quest(mission_id)
    rel = (quest.get("patch") or {}).get("starter_file")
    if not isinstance(rel, str) or not rel:
        raise FileNotFoundError(f"{mission_id} has no patch.starter_file in quests yaml")
    resolved = (ROOT / rel).resolve()
    repo_root = ROOT.resolve()
    if not str(resolved).startswith(str(repo_root) + "/"):
        raise PermissionError(f"starter path escapes repo: {rel}")
    if "/labs/" not in str(resolved):
        raise PermissionError(f"refusing to edit non-lab path: {rel}")
    return resolved


def starter_payload(mission_id: str) -> dict[str, Any]:
    quest = get_quest(mission_id)
    patch = quest.get("patch") or {}
    rel = patch.get("starter_file")
    if not isinstance(rel, str) or not rel:
        raise FileNotFoundError(f"{mission_id} has no patch.starter_file")
    path = _starter_path(mission_id)
    if not path.exists():
        return {
            "mission": mission_id,
            "path": rel,
            "language": rel.rsplit(".", 1)[-1] if "." in rel else "text",
            "content": "",
            "exists": False,
        }
    content = path.read_text(encoding="utf-8")
    return {
        "mission": mission_id,
        "path": rel,
        "language": rel.rsplit(".", 1)[-1] if "." in rel else "text",
        "content": content,
        "exists": True,
        "size": len(content.encode("utf-8")),
    }


def write_starter(mission_id: str, content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if len(content.encode("utf-8")) > STARTER_BYTE_LIMIT:
        raise ValueError(f"content exceeds {STARTER_BYTE_LIMIT} bytes")
    path = _starter_path(mission_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return starter_payload(mission_id)


def patch_payload(mission_id: str) -> dict[str, Any]:
    """Everything the mission-detail UI needs to render the Patch Track:
    - task.md content (rendered as markdown by frontend)
    - status (PASS/FAIL/never run)
    - file paths for starter / reference / tests
    """
    quest = get_quest(mission_id)
    patch = quest.get("patch")
    if not patch:
        raise FileNotFoundError(
            f"{mission_id} has no `patch:` block in quests/{mission_id}.yaml"
        )

    def _read(rel: str | None) -> str | None:
        if not rel:
            return None
        path = ROOT / rel
        return path.read_text(encoding="utf-8") if path.exists() else None

    status_path = _resolve_patch_status_path(quest)
    status = _read_patch_status_file(status_path)

    quiz_state = quiz_status(mission_id)
    quiz_passed = bool(quiz_state.get("passed"))

    return {
        "mission": mission_id,
        "title": quest.get("title"),
        "description": patch.get("description"),
        "test_command": patch.get("test_command"),
        "test_count": patch.get("test_count"),
        "test_kind": patch.get("test_kind"),
        "task_md_path": patch.get("task_md"),
        "task_md_content": _read(patch.get("task_md")),
        "starter_file": patch.get("starter_file"),
        "reference_file": patch.get("reference_file"),
        "status": status,  # null until first run
        "quiz_passed": quiz_passed,  # ★ gate signal for the frontend
        "lesson": quest.get("lesson"),
        "lesson_docs": quest.get("lesson_docs", []),
        "source_reading": quest.get("source_reading", []),
        "notebooks": quest.get("notebooks", []),
        "mini_infra_targets": quest.get("mini_infra_targets", []),
        "tickets": quest.get("tickets", []),
        "prereq": quest.get("prereq"),
        "next_lab": quest.get("next_lab"),
        "estimated_minutes": quest.get("estimated_minutes"),
        "no_gpu_friendly": quest.get("no_gpu_friendly", False),
    }


def list_prompt_cards() -> list[dict[str, Any]]:
    cards = []
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first_line = next(
            (line for line in text.splitlines() if line.strip()), path.stem
        )
        cards.append(
            {
                "id": path.stem,
                "title": first_line.lstrip("# ").strip(),
                "body": text,
                "path": str(path.relative_to(ROOT)),
            }
        )
    return cards


def mission_runs(mission_id: str) -> list[Path]:
    directory = RUNS_DIR / mission_id
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_dir()])


def latest_run(mission_id: str) -> Path | None:
    runs = mission_runs(mission_id)
    return runs[-1] if runs else None


def summarize_run(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    return {
        "run_id": path.name,
        "path": str(path.relative_to(ROOT)),
        "metrics_count": len(metrics),
        "latest_metric": metrics[-1] if metrics else None,
    }


def badge_status() -> list[dict[str, Any]]:
    """PASS/FAIL is now driven by patch-test result, not grade.json."""
    rows = []
    for quest in list_quests():
        status_path = _resolve_patch_status_path(quest)
        status = _read_patch_status_file(status_path)
        rows.append(
            {
                "mission": quest["id"],
                "level": quest.get("level"),
                "title": quest.get("title"),
                "passed": bool(status and status.get("passed")),
                "test_count": (quest.get("patch") or {}).get("test_count"),
                "summary": status.get("summary") if status else None,
                "finished_at": status.get("finished_at") if status else None,
            }
        )
    return rows


def dashboard_payload() -> dict[str, Any]:
    quests = list_quests()
    rows = []
    completed = 0
    for quest in quests:
        status_path = _resolve_patch_status_path(quest)
        status = _read_patch_status_file(status_path)
        passed = bool(status and status.get("passed"))
        completed += int(passed)
        patch = quest.get("patch") or {}
        rows.append(
            {
                "id": quest["id"],
                "title": quest["title"],
                "level": quest.get("level"),
                "role": quest.get("role"),
                "frameworks": quest.get("frameworks", []),
                "gpu_modes": quest.get("gpu_modes", {}),
                "estimated_minutes": quest.get("estimated_minutes"),
                "no_gpu_friendly": quest.get("no_gpu_friendly", False),
                "patch_test_command": patch.get("test_command"),
                "patch_status": status,  # null if never run
                "status": "已通过" if passed else (
                    "失败重试" if status and not passed else "未开始"
                ),
            }
        )
    current = next((row for row in rows if row["status"] != "已通过"), None)
    return {
        "current_level": current["level"] if current else None,
        "current_role": current["role"] if current else None,
        "active_mission": current["id"] if current else None,
        "completed_missions": completed,
        "total_missions": len(quests),
        "missions": rows,
    }


SELF_STUDY_LOOP = [
    {
        "name": "读本关讲义",
        "duration": "15-30 分钟",
        "action": "先像上课一样读懂背景，只在讲义提示时看高亮源码。",
    },
    {
        "name": "做小实验 / 对照源码",
        "duration": "20-40 分钟",
        "action": "用 notebook 验证讲义里的直觉，再点源码片段跳到完整文件。",
    },
    {
        "name": "做 Quiz，再写 Lab",
        "duration": "60-180 分钟",
        "action": "Quiz 只检查讲义关键概念；通过后在 patch/starter/ 里实现讲义的核心小机制。",
    },
    {
        "name": "跑最后验证命令",
        "duration": "1-3 分钟",
        "action": "make patch-test M=<lab> —— pytest 全绿说明这个小机制实现正确。",
    },
    {
        "name": "AI 框架理解口试",
        "duration": "20-40 分钟",
        "action": "用 prompts/framework_understanding_tutor.md 提问，直到能讲清 MiniInfra → 真实源码 → debug ticket 的连接。",
    },
    {
        "name": "（可选）做 ticket",
        "duration": "30-60 分钟",
        "action": "选一个 debug ticket，按最小复现定位。",
    },
]

STUCK_PLAYBOOK = [
    {
        "symptom": "看不懂 task.md 的接口契约",
        "action": "先打开对应 notebook（见 task.md 顶部链接）把概念图画一遍。",
    },
    {
        "symptom": "starter 里的 TODO 不知道怎么填",
        "action": "make patch-hint M=<lab> —— 看 TODO 列表 + 关键提示。",
    },
    {
        "symptom": "跑 patch-test 报形状错配 / 数值不对",
        "action": "用 print 在 forward 里打张量 shape、dtype、device；与 task.md 的契约逐项核对。",
    },
    {
        "symptom": "30+ 分钟仍卡住",
        "action": "make patch-show-solution M=<lab> 看参考解，但建议关掉答案再自己写一遍。",
    },
    {
        "symptom": "patch-test 通过了但想精进",
        "action": "打开 github_repo/<framework>/ 的对应文件，看真实工程版多了哪些边界。",
    },
]

COMPLETION_CHECKS = [
    "已经读完本关讲义，并能说出每段源码穿插在实现哪个概念。",
    "patch/starter/<file>.py 的所有 TODO 都填实，无 NotImplementedError。",
    "make patch-test M=<lab> 全绿（pytest 0 failed）。",
    "能讲清楚自己 patch 的核心 abstraction 是什么、哪一步必须通信 / 不能通信。",
    "能找出 github_repo/ 真实框架版与自己的 patch 在工程边界上的 3 处不同。",
    "能通过 docs/AI框架理解评估指南.md 定义的五层 AI 口试。",
]


def _recommended_command(quest: dict[str, Any]) -> str | None:
    """Now points at patch-test as the single recommended action per lab."""
    patch = quest.get("patch") or {}
    return patch.get("test_command")


def _doc_card(path: Path) -> dict[str, str]:
    body = path.read_text(encoding="utf-8") if path.exists() else ""
    title = next(
        (
            line.lstrip("# ").strip()
            for line in body.splitlines()
            if line.startswith("#")
        ),
        path.stem,
    )
    summary = next(
        (
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.startswith("#")
        ),
        "",
    )
    return {"title": title, "path": str(path.relative_to(ROOT)), "summary": summary}


def curriculum_payload() -> dict[str, Any]:
    quests = list_quests()
    phases: list[dict[str, Any]] = []
    phase_lookup: dict[str, dict[str, Any]] = {}
    mission_cards = []
    completed_count = 0

    for quest in quests:
        status_path = _resolve_patch_status_path(quest)
        status = _read_patch_status_file(status_path)
        passed = bool(status and status.get("passed"))
        completed_count += int(passed)
        recommended = _recommended_command(quest)
        source_reading = quest.get("source_reading", [])
        notebooks = quest.get("notebooks", [])
        tickets = quest.get("tickets", [])
        patch = quest.get("patch") or {}
        card = {
            "id": quest["id"],
            "level": quest.get("level"),
            "title": quest.get("title"),
            "act": quest.get("act"),
            "role": quest.get("role"),
            "frameworks": quest.get("frameworks", []),
            "patch_description": patch.get("description"),
            "patch_test_count": patch.get("test_count"),
            "source_count": len(source_reading),
            "lesson_section_count": len((quest.get("lesson") or {}).get("sections", [])),
            "notebook_count": len(notebooks),
            "ticket_count": len(tickets),
            "recommended_command": recommended,
            "status": "已通过" if passed else (
                "失败重试" if status else "未开始"
            ),
            "patch_status": status,
        }
        mission_cards.append(card)
        phase_key = quest.get("act") or "未归类"
        if phase_key not in phase_lookup:
            phase_lookup[phase_key] = {
                "name": phase_key,
                "missions": [],
                "frameworks": [],
                "source_count": 0,
                "notebook_count": 0,
                "completed_missions": 0,
            }
            phases.append(phase_lookup[phase_key])
        phase = phase_lookup[phase_key]
        phase["missions"].append(card)
        phase["source_count"] += len(source_reading)
        phase["notebook_count"] += len(notebooks)
        phase["completed_missions"] += int(passed)
        for framework in quest.get("frameworks", []):
            if framework not in phase["frameworks"]:
                phase["frameworks"].append(framework)

    doc_files = [
        DOCS_DIR / "自学使用指南.md",
        DOCS_DIR / "能力矩阵.md",
        DOCS_DIR / "AI框架理解评估指南.md",
        DOCS_DIR / "mini_infra.md",
        DOCS_DIR / "硬件路径矩阵.md",
        DOCS_DIR / "术语表.md",
        DOCS_DIR / "源码研读地图.md",
        NOTEBOOKS_DIR / "README.md",
    ]
    docs = [_doc_card(p) for p in doc_files if p.exists()]
    current = next(
        (mission for mission in mission_cards if mission["status"] != "已通过"), None
    )
    return {
        "title": "Infra Quest 自学路线",
        "principle": "单人自学优先：先读本关讲义，再看穿插源码，最后用 Quiz 和 Patch 验证是否真的懂。",
        "stats": {
            "total_missions": len(mission_cards),
            "completed_missions": completed_count,
            "total_source_nodes": sum(
                mission["source_count"] for mission in mission_cards
            ),
            "total_notebooks": len(
                {
                    notebook
                    for quest in quests
                    for notebook in quest.get("notebooks", [])
                }
            ),
            "total_projects": len(
                [mission for mission in mission_cards if mission.get("project_name")]
            ),
        },
        "current_mission": current,
        "self_study_loop": SELF_STUDY_LOOP,
        "stuck_playbook": STUCK_PLAYBOOK,
        "completion_checks": COMPLETION_CHECKS,
        "docs": docs,
        "phases": phases,
        "missions": mission_cards,
    }


def report_payload(mission_id: str) -> dict[str, Any]:
    run = latest_run(mission_id)
    if run and (run / "report.md").exists():
        return {
            "mission": mission_id,
            "source": str((run / "report.md").relative_to(ROOT)),
            "body": (run / "report.md").read_text(encoding="utf-8"),
        }
    report_path = REPORTS_DIR / f"{mission_id}.md"
    if report_path.exists():
        return {
            "mission": mission_id,
            "source": str(report_path.relative_to(ROOT)),
            "body": report_path.read_text(encoding="utf-8"),
        }
    return {
        "mission": mission_id,
        "source": None,
        "body": "# 暂无报告\n\n请先运行 smoke 并完成 report.md。\n",
    }


def metrics_payload(mission_id: str | None = None) -> list[dict[str, Any]]:
    payload = []
    missions = [mission_id] if mission_id else [quest["id"] for quest in list_quests()]
    for mission in missions:
        run = latest_run(mission)
        if not run:
            continue
        for row in read_jsonl(run / "metrics.jsonl"):
            payload.append({"mission": mission, "run_id": run.name, **row})
    return payload


def concept_map_payload() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in CONCEPT_MAP.items()]


def source_map_payload(framework: str) -> dict[str, Any]:
    key = framework
    if framework.lower() == "megatron":
        key = "Megatron"
    if framework.lower() == "torchtitan":
        key = "TorchTitan"
    if framework.lower() == "slime":
        key = "SLiME"
    matched = next((name for name in SOURCE_MAPS if name.lower() == key.lower()), None)
    if not matched:
        raise FileNotFoundError(framework)
    return {"framework": matched, "nodes": SOURCE_MAPS[matched]}


def _safe_notebook_path(notebook_path: str) -> Path:
    normalized = notebook_path.strip().lstrip("/")
    if normalized.startswith("notebooks/"):
        normalized = normalized.removeprefix("notebooks/")
    candidate = (NOTEBOOKS_DIR / normalized).resolve()
    if NOTEBOOKS_DIR.resolve() not in candidate.parents or candidate.suffix != ".ipynb":
        raise FileNotFoundError(notebook_path)
    if not candidate.exists():
        raise FileNotFoundError(notebook_path)
    return candidate


SOURCE_ALLOWED_ROOTS: tuple[str, ...] = (
    "github_repo",
    "labs",
    "scripts",
    "docs",
    "notebooks",
    "quests",
    "tickets",
    "prompts",
    "app",
    "autograder",
    "mini_infra",
    "simulators",
    "envs",
    "dashboards",
    "final_artifacts",
)

SOURCE_MAX_BYTES = 2 * 1024 * 1024  # 2 MB

SOURCE_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "json",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hh": "cpp",
    ".cu": "cuda-cpp",
    ".cuh": "cuda-cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".proto": "proto",
    ".sql": "sql",
    ".dockerfile": "dockerfile",
    ".ipynb": "json",
    ".txt": "text",
    ".env": "bash",
    ".lock": "text",
    ".gitignore": "text",
    ".gitattributes": "text",
    ".mk": "makefile",
    ".make": "makefile",
    ".m": "objc",
    ".mm": "objc",
}

SOURCE_TEXT_EXTENSIONS = set(SOURCE_LANGUAGE_BY_EXT.keys()) | {".tsv", ".csv", ".log"}

SOURCE_FILENAME_LANG: dict[str, str] = {
    "makefile": "makefile",
    "dockerfile": "dockerfile",
}

SOURCE_ASSET_MEDIA_TYPES: dict[str, str] = {
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_source_path(rel_path: str) -> Path:
    if rel_path is None:
        raise FileNotFoundError(rel_path)
    cleaned = rel_path.strip().lstrip("/")
    if not cleaned:
        raise FileNotFoundError(rel_path)
    if any(part == ".." for part in Path(cleaned).parts):
        raise PermissionError(rel_path)
    first_segment = cleaned.split("/", 1)[0]
    if first_segment not in SOURCE_ALLOWED_ROOTS:
        raise PermissionError(rel_path)
    candidate = (ROOT / cleaned).resolve()
    if not _is_within_root(candidate, ROOT.resolve()):
        raise PermissionError(rel_path)
    if not candidate.exists():
        raise FileNotFoundError(rel_path)
    return candidate


def _detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SOURCE_LANGUAGE_BY_EXT:
        return SOURCE_LANGUAGE_BY_EXT[suffix]
    name = path.name.lower()
    if name in SOURCE_FILENAME_LANG:
        return SOURCE_FILENAME_LANG[name]
    return "text"


def _is_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in SOURCE_TEXT_EXTENSIONS:
        return True
    if path.name.lower() in SOURCE_FILENAME_LANG:
        return True
    if suffix == "":
        try:
            sample = path.read_bytes()[:4096]
        except OSError:
            return False
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


def source_payload(rel_path: str) -> dict[str, Any]:
    path = _safe_source_path(rel_path)
    if path.is_dir():
        raise IsADirectoryError(rel_path)
    if not _is_text_file(path):
        raise PermissionError(
            f"binary or unsupported file type: {path.suffix or path.name}"
        )
    size = path.stat().st_size
    if size > SOURCE_MAX_BYTES:
        return {
            "path": str(path.relative_to(ROOT)),
            "language": _detect_language(path),
            "size": size,
            "line_count": 0,
            "content": "",
            "truncated": True,
            "reason": f"file exceeds {SOURCE_MAX_BYTES} bytes",
        }
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PermissionError(f"non-utf8 binary file: {path.name}") from exc
    if content == "":
        line_count = 0
    elif content.endswith("\n"):
        line_count = content.count("\n")
    else:
        line_count = content.count("\n") + 1
    return {
        "path": str(path.relative_to(ROOT)),
        "language": _detect_language(path),
        "size": size,
        "line_count": line_count,
        "content": content,
        "truncated": False,
    }


def source_asset_path(rel_path: str) -> tuple[Path, str]:
    path = _safe_source_path(rel_path)
    if path.is_dir():
        raise IsADirectoryError(rel_path)
    media_type = SOURCE_ASSET_MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        raise PermissionError(
            f"binary or unsupported asset type: {path.suffix or path.name}"
        )
    return path, media_type


def source_tree_payload(rel_dir: str) -> dict[str, Any]:
    cleaned = rel_dir.strip().lstrip("/") if rel_dir else ""
    if not cleaned:
        entries = []
        for root in SOURCE_ALLOWED_ROOTS:
            p = ROOT / root
            if p.exists():
                entries.append(
                    {
                        "name": root,
                        "path": root,
                        "type": "dir" if p.is_dir() else "file",
                        "size": None,
                    }
                )
        return {"path": "", "entries": entries}
    path = _safe_source_path(cleaned)
    if not path.is_dir():
        raise NotADirectoryError(cleaned)
    entries = []
    for child in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if child.name.startswith("."):
            continue
        if child.is_file() and not _is_text_file(child):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(ROOT)),
                    "type": "binary",
                    "size": child.stat().st_size,
                }
            )
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(ROOT)),
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return {"path": str(path.relative_to(ROOT)), "entries": entries}


def notebook_payload(notebook_path: str) -> dict[str, Any]:
    path = _safe_notebook_path(notebook_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    title = path.stem
    for index, cell in enumerate(payload.get("cells", [])):
        source = "".join(cell.get("source", []))
        if index == 0 and cell.get("cell_type") == "markdown":
            first_heading = next(
                (
                    line.lstrip("# ").strip()
                    for line in source.splitlines()
                    if line.startswith("#")
                ),
                None,
            )
            title = first_heading or title
        outputs = []
        for output in cell.get("outputs", []):
            if "text" in output:
                outputs.append(
                    {"type": "text", "body": "".join(output.get("text", []))}
                )
            elif "data" in output:
                data = output.get("data", {})
                if "text/plain" in data:
                    outputs.append(
                        {"type": "text", "body": "".join(data.get("text/plain", []))}
                    )
                elif "image/png" in data:
                    outputs.append({"type": "image/png", "body": data.get("image/png")})
        cells.append(
            {
                "index": index,
                "cell_type": cell.get("cell_type", "raw"),
                "source": source,
                "execution_count": cell.get("execution_count"),
                "outputs": outputs,
            }
        )
    return {
        "path": f"notebooks/{path.name}",
        "title": title,
        "cell_count": len(cells),
        "cells": cells,
    }
