from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.backend.repository import (
    badge_status,
    concept_map_payload,
    curriculum_payload,
    dashboard_payload,
    get_quest,
    get_ticket,
    list_prompt_cards,
    list_quests,
    list_tickets,
    metrics_payload,
    notebook_payload,
    patch_payload,
    patch_status,
    quiz_payload,
    quiz_status,
    report_payload,
    source_map_payload,
    source_payload,
    source_tree_payload,
    submit_quiz,
)

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Infra Quest Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return dashboard_payload()


@app.get("/api/quests")
def quests() -> list[dict[str, Any]]:
    return list_quests()


@app.get("/api/curriculum")
def curriculum() -> dict[str, Any]:
    return curriculum_payload()


@app.get("/api/quests/{mission_id}")
def quest_detail(mission_id: str) -> dict[str, Any]:
    try:
        return get_quest(mission_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# ★ Quiz Gate endpoints — must pass before patch test counts.
# ---------------------------------------------------------------------------


@app.get("/api/missions/{mission_id}/quiz")
def quiz_get(mission_id: str) -> dict[str, Any]:
    try:
        return quiz_payload(mission_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/missions/{mission_id}/quiz")
def quiz_submit(mission_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(status_code=400, detail="answers must be a dict")
    try:
        return submit_quiz(mission_id, answers)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/missions/{mission_id}/quiz-status")
def quiz_status_get(mission_id: str) -> dict[str, Any]:
    return quiz_status(mission_id)


# ---------------------------------------------------------------------------
# ★ Patch Track endpoints — the single most important integration point.
# ---------------------------------------------------------------------------


@app.get("/api/missions/{mission_id}/patch")
def patch_info(mission_id: str) -> dict[str, Any]:
    try:
        return patch_payload(mission_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/missions/{mission_id}/patch-test")
async def run_patch_test(mission_id: str) -> dict[str, Any]:
    """Trigger `python scripts/run_patch_test.py --mission <id>` and return
    the resulting status JSON. Streaming output is intentionally NOT implemented
    here — the runner writes a status file and we read it after completion.
    For a 15–60 s patch test this is acceptable; we can upgrade to SSE later."""
    runner = ROOT / "scripts" / "run_patch_test.py"
    if not runner.exists():
        raise HTTPException(status_code=500, detail=f"runner missing: {runner}")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(runner),
        "--mission",
        mission_id,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout_bytes, _ = await proc.communicate()
    # The runner persists the canonical result; just read it back.
    try:
        return patch_status(mission_id)
    except FileNotFoundError:
        # Runner didn't even produce a status file → return raw stdout for diagnosis.
        return {
            "mission": mission_id,
            "passed": False,
            "exit_code": proc.returncode,
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "output": stdout_bytes.decode("utf-8", errors="replace"),
            "error": "runner did not produce .last_run.json",
        }


@app.get("/api/missions/{mission_id}/patch-status")
def patch_status_only(mission_id: str) -> dict[str, Any]:
    try:
        return patch_status(mission_id)
    except FileNotFoundError:
        return {"mission": mission_id, "passed": False, "never_run": True}


# ---------------------------------------------------------------------------
# Existing read-only endpoints
# ---------------------------------------------------------------------------


@app.get("/api/tickets")
def tickets() -> list[dict[str, Any]]:
    return list_tickets()


@app.get("/api/tickets/{ticket_id}")
def ticket_detail(ticket_id: str) -> dict[str, Any]:
    try:
        return get_ticket(ticket_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/metrics")
def metrics(mission: str | None = None) -> list[dict[str, Any]]:
    return metrics_payload(mission)


@app.get("/api/reports/{mission_id}")
def reports(mission_id: str) -> dict[str, Any]:
    return report_payload(mission_id)


@app.get("/api/badges")
def badges() -> list[dict[str, Any]]:
    return badge_status()


@app.get("/api/prompts")
def prompts() -> list[dict[str, Any]]:
    return list_prompt_cards()


@app.get("/api/notebooks")
def notebook(path: str) -> dict[str, Any]:
    try:
        return notebook_payload(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/concept-map")
def concept_map() -> list[dict[str, Any]]:
    return concept_map_payload()


@app.get("/api/source-map/{framework}")
def source_map(framework: str) -> dict[str, Any]:
    try:
        return source_map_payload(framework)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/source")
def source(path: str) -> dict[str, Any]:
    try:
        return source_payload(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IsADirectoryError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"path is a directory; use /api/source/tree?dir={path}",
        ) from exc


@app.get("/api/source/tree")
def source_tree(dir: str = "") -> dict[str, Any]:
    try:
        return source_tree_payload(dir)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"path is a file; use /api/source?path={dir}",
        ) from exc
