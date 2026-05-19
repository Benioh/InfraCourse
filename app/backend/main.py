from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.backend.ai_client import build_system_prompt, stream_response
from app.backend.ai_config import all_providers
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
    starter_payload,
    submit_quiz,
    write_starter as repo_write_starter,
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
# ★ Starter file endpoints — direct in-app editing.
# ---------------------------------------------------------------------------


@app.get("/api/missions/{mission_id}/starter")
def starter_get(mission_id: str) -> dict[str, Any]:
    try:
        return starter_payload(mission_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.put("/api/missions/{mission_id}/starter")
def starter_put(mission_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content (string) required")
    try:
        return repo_write_starter(mission_id, content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# ★ Starter file editing — backs the in-app editor + the AI's write_starter
# tool. Path is resolved from quests/<id>.yaml > patch.starter_file and is
# constrained to live under labs/<mission>/patch/starter/.
# ---------------------------------------------------------------------------


def _starter_path(mission_id: str) -> Path:
    quest = get_quest(mission_id)
    rel = (quest.get("patch") or {}).get("starter_file")
    if not isinstance(rel, str) or not rel:
        raise HTTPException(
            status_code=404,
            detail=f"{mission_id} has no patch.starter_file in quests yaml",
        )
    target = (ROOT / rel).resolve()
    expected_root = (ROOT / "labs" / mission_id / "patch" / "starter").resolve()
    if target != expected_root and not str(target).startswith(str(expected_root) + "/"):
        raise HTTPException(
            status_code=403,
            detail=f"starter file {rel!r} escapes labs/{mission_id}/patch/starter/",
        )
    return target


@app.get("/api/missions/{mission_id}/starter")
def starter_get(mission_id: str) -> dict[str, Any]:
    path = _starter_path(mission_id)
    if not path.exists():
        return {
            "mission": mission_id,
            "path": str(path.relative_to(ROOT)),
            "content": "",
            "exists": False,
        }
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail=f"binary starter: {exc}") from exc
    return {
        "mission": mission_id,
        "path": str(path.relative_to(ROOT)),
        "content": content,
        "exists": True,
        "bytes": len(content.encode("utf-8")),
    }


@app.put("/api/missions/{mission_id}/starter")
def starter_put(mission_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content (string) required")
    if len(content.encode("utf-8")) > 200_000:
        raise HTTPException(status_code=413, detail="content exceeds 200KB")
    path = _starter_path(mission_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "mission": mission_id,
        "path": str(path.relative_to(ROOT)),
        "bytes": len(content.encode("utf-8")),
        "saved": True,
    }


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


# ---------------------------------------------------------------------------
# AI tutor — zero-config provider switch backed by ~/.claude and ~/.codex.
# ---------------------------------------------------------------------------


@app.get("/api/ai/providers")
def ai_providers() -> dict[str, list[str]]:
    """Which AI providers the backend has working credentials for."""
    return {"available": sorted(all_providers().keys())}


@app.post("/api/ai/ask")
async def ai_ask(payload: dict[str, Any]) -> StreamingResponse:
    """SSE stream of normalized AI tutor events. See ai_client.stream_response."""
    provider = payload.get("provider") or "claude"
    if provider not in {"claude", "codex"}:
        raise HTTPException(status_code=400, detail="provider must be 'claude' or 'codex'")
    mode = payload.get("mode") or "tutor"
    if mode not in {"tutor", "coder"}:
        raise HTTPException(status_code=400, detail="mode must be 'tutor' or 'coder'")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="message required")
    context = payload.get("context") or {}
    if not isinstance(context, dict):
        raise HTTPException(status_code=400, detail="context must be an object")
    history = payload.get("history") or []
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be a list")
    if provider not in all_providers():
        raise HTTPException(
            status_code=503,
            detail=f"provider {provider!r} not configured on this server",
        )

    system = build_system_prompt(context, mode=mode)

    async def event_stream() -> Any:
        try:
            async for event in stream_response(
                provider, system, history, message.strip(), mode=mode
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected — let the generator unwind cleanly.
            raise
        except Exception as exc:  # noqa: BLE001 - last-line surface to client
            err = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
