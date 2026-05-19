"""Tools the AI tutor can call.

Tutor mode: navigate_to_source, read_file, search_code (read-only).
Vibe-coding mode: tutor tools + write_starter, run_patch_test (editing the
mission's starter file and triggering its patch test).

All paths flow through `repository._safe_source_path` so they share the same
sandbox as the existing `/api/source` endpoint. `search_code` shells out to
ripgrep; if `rg` is missing the tool returns an error rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.backend.repository import (
    ROOT,
    SOURCE_ALLOWED_ROOTS,
    _safe_source_path,
    get_quest,
    patch_status,
    source_payload,
)

logger = logging.getLogger(__name__)


TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "navigate_to_source",
        "description": (
            "Open a source file in the in-app viewer at a specific line range. "
            "Use when the learner wants to jump to a definition or related code."
        ),
        "input_schema": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Repo-relative path under one of: github_repo/, labs/, "
                        "mini_infra/, scripts/, docs/, notebooks/, app/, etc."
                    ),
                },
                "lines": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[start, end] inclusive, 1-indexed.",
                },
            },
        },
    },
    {
        "name": "read_file",
        "description": (
            "Return the contents of a source file. Optionally restrict to a line "
            "range to keep the response small. Call this before answering questions "
            "about specific code that is not already in the conversation."
        ),
        "input_schema": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
            },
        },
    },
    {
        "name": "search_code",
        "description": (
            "ripgrep across the course source roots. Use to find symbol definitions, "
            "call sites, or string occurrences. Returns up to 50 hits."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "path_glob": {
                    "type": "string",
                    "description": "rg --glob filter, e.g. '*.py' or '**/scheduler.py'",
                },
                "regex": {
                    "type": "boolean",
                    "default": False,
                    "description": "If false (default), the query is matched literally.",
                },
            },
        },
    },
]


SEARCH_HIT_CAP = 50
SEARCH_TIMEOUT_S = 15
_REGEX_METACHARS = re.compile(r"[\^$.|?*+()\[\]{}\\]")


def _source_url(path: str, lines: list[int] | None) -> str:
    if not lines:
        return f"/source?path={path}"
    a, b = lines
    fragment = f"{a}" if a == b else f"{a}-{b}"
    return f"/source?path={path}&lines={fragment}"


def navigate_to_source(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("path")
    if not isinstance(raw, str):
        return {"error": "path must be a string"}
    try:
        resolved = _safe_source_path(raw)
    except (PermissionError, FileNotFoundError) as exc:
        return {"error": f"path rejected: {exc}"}
    if resolved.is_dir():
        return {"error": "path is a directory; use search_code or list a file instead"}
    rel = str(resolved.relative_to(ROOT))

    raw_lines = args.get("lines")
    lines_clean: list[int] | None = None
    if (
        isinstance(raw_lines, list)
        and len(raw_lines) == 2
        and all(isinstance(n, int) and n > 0 for n in raw_lines)
    ):
        a, b = int(raw_lines[0]), int(raw_lines[1])
        lines_clean = [min(a, b), max(a, b)]

    return {
        "path": rel,
        "lines": lines_clean,
        "url": _source_url(rel, lines_clean),
    }


def read_file(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("path")
    if not isinstance(raw, str):
        return {"error": "path must be a string"}
    try:
        payload = source_payload(raw)
    except (PermissionError, FileNotFoundError, IsADirectoryError) as exc:
        return {"error": f"read_file rejected: {exc}"}
    if payload.get("truncated"):
        return payload

    line_start = args.get("line_start")
    line_end = args.get("line_end")
    if not isinstance(line_start, int) and not isinstance(line_end, int):
        return payload

    content: str = payload.get("content", "")
    line_count: int = payload.get("line_count", 0) or 0
    all_lines = content.splitlines()
    a = max(1, int(line_start)) if isinstance(line_start, int) else 1
    b = int(line_end) if isinstance(line_end, int) else line_count
    if b < a:
        a, b = b, a
    b = min(b, len(all_lines))
    a = min(a, b) if b else a
    sliced = all_lines[a - 1 : b] if b >= a >= 1 else []
    return {
        **payload,
        "content": "\n".join(sliced),
        "line_start": a,
        "line_end": b,
        "sliced": True,
    }


def _looks_like_regex(query: str) -> bool:
    return bool(_REGEX_METACHARS.search(query))


_RG_BIN: str | None = shutil.which("rg")


def search_code(args: dict[str, Any]) -> dict[str, Any]:
    if _RG_BIN is None:
        return {"error": "ripgrep (rg) is not installed on the server"}
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "query must be a non-empty string"}

    use_regex = bool(args.get("regex"))
    glob = args.get("path_glob") if isinstance(args.get("path_glob"), str) else None

    cmd: list[str] = [
        _RG_BIN,
        "--json",
        "-n",
        "--max-count",
        "3",
        "--max-columns",
        "400",
    ]
    if glob:
        cmd.extend(["--glob", glob])
    if use_regex:
        cmd.append(query)
    else:
        cmd.extend(["-F", query])

    roots = [str(ROOT / r) for r in SOURCE_ALLOWED_ROOTS if (ROOT / r).exists()]
    if not roots:
        return {"error": "no source roots available"}
    cmd.extend(roots)

    try:
        proc = subprocess.run(  # noqa: S603 - args list, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT_S,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"error": "search timed out"}

    hits: list[dict[str, Any]] = []
    truncated = False
    root_resolved = ROOT.resolve()
    for line in proc.stdout.splitlines():
        if len(hits) >= SEARCH_HIT_CAP:
            truncated = True
            break
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "match":
            continue
        d = ev.get("data") or {}
        path_text = (d.get("path") or {}).get("text") or ""
        try:
            rel = str(Path(path_text).resolve().relative_to(root_resolved))
        except (ValueError, OSError):
            continue
        line_num = d.get("line_number")
        preview = ((d.get("lines") or {}).get("text") or "").rstrip("\n")
        hits.append({"path": rel, "line": line_num, "preview": preview[:200]})

    return {
        "query": query,
        "regex": use_regex,
        "glob": glob,
        "hits": hits,
        "count": len(hits),
        "truncated": truncated,
    }


# ----------------------------------------------------------------------------
# Vibe-coding mode: write_starter + run_patch_test
#
# The vibe-coding doc forbids the AI from autonomously running patch-test, so
# we still expose `run_patch_test` as a tool — but the system prompt tells the
# model to only call it when the learner explicitly asks. write_starter is the
# only edit-side primitive; it can only touch the mission's starter_file
# declared in `quests/<id>.yaml`.
# ----------------------------------------------------------------------------


PATCH_TEST_TIMEOUT_S = 180
STARTER_BYTE_LIMIT = 200_000


def _starter_path_for(mission_id: str) -> Path:
    quest = get_quest(mission_id)
    rel = (quest.get("patch") or {}).get("starter_file")
    if not isinstance(rel, str) or not rel:
        raise FileNotFoundError(f"{mission_id} has no patch.starter_file in quests yaml")
    resolved = (ROOT / rel).resolve()
    if not str(resolved).startswith(str(ROOT.resolve()) + "/"):
        raise PermissionError(f"starter path escapes repo: {rel}")
    if "/labs/" not in str(resolved):
        raise PermissionError(f"refusing to edit non-lab path: {rel}")
    return resolved


def write_starter(args: dict[str, Any]) -> dict[str, Any]:
    mission = args.get("mission")
    if not isinstance(mission, str) or not mission:
        return {"error": "mission required"}
    content = args.get("content")
    if not isinstance(content, str):
        return {"error": "content (string) required"}
    if len(content.encode("utf-8")) > STARTER_BYTE_LIMIT:
        return {"error": f"content exceeds {STARTER_BYTE_LIMIT} bytes"}
    stage = args.get("stage")
    note = args.get("note")
    try:
        path = _starter_path_for(mission)
    except (FileNotFoundError, PermissionError) as exc:
        return {"error": str(exc)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    rel = str(path.relative_to(ROOT))
    return {
        "mission": mission,
        "path": rel,
        "bytes": len(content.encode("utf-8")),
        "stage": stage if isinstance(stage, (int, str)) else None,
        "note": note if isinstance(note, str) else None,
    }


def run_patch_test(args: dict[str, Any]) -> dict[str, Any]:
    mission = args.get("mission")
    if not isinstance(mission, str) or not mission:
        return {"error": "mission required"}
    runner = ROOT / "scripts" / "run_patch_test.py"
    if not runner.exists():
        return {"error": f"runner missing: {runner}"}
    try:
        proc = subprocess.run(  # noqa: S603 - args list, no shell
            [sys.executable, str(runner), "--mission", mission],
            capture_output=True,
            text=True,
            timeout=PATCH_TEST_TIMEOUT_S,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"patch test timed out after {PATCH_TEST_TIMEOUT_S}s", "mission": mission}
    try:
        status = patch_status(mission)
    except FileNotFoundError:
        return {
            "mission": mission,
            "passed": False,
            "exit_code": proc.returncode,
            "output_tail": (proc.stdout or proc.stderr or "")[-2000:],
            "error": "runner did not produce .last_run.json",
        }
    out = status.get("output") or ""
    return {
        **{k: v for k, v in status.items() if k != "output"},
        "output_tail": out[-3000:] if isinstance(out, str) else "",
    }


VIBECODING_TOOL_SCHEMA: list[dict[str, Any]] = TOOL_SCHEMA + [
    {
        "name": "write_starter",
        "description": (
            "Replace the contents of the current mission's `patch.starter_file`. "
            "Use ONLY after the learner has confirmed (a) lab id, (b) patch "
            "contract, (c) source map paths, and (d) their own one-line guess. "
            "Always say which stage (1=skeleton, 2=core algo, 3=boundary) you "
            "are committing, and what this stage does NOT cover."
        ),
        "input_schema": {
            "type": "object",
            "required": ["mission", "content"],
            "properties": {
                "mission": {"type": "string", "description": "Mission id, e.g. l03_nccl_ddp_smoke."},
                "content": {"type": "string", "description": "Full new contents of starter_file."},
                "stage": {
                    "type": ["integer", "string"],
                    "description": "Stage label (1/2/3 or 'skeleton'/'core'/'boundary').",
                },
                "note": {
                    "type": "string",
                    "description": "One-line summary of what this stage covers / does not cover.",
                },
            },
        },
    },
    {
        "name": "run_patch_test",
        "description": (
            "Run `make patch-test` for a mission and return PASS/FAIL plus tail "
            "of pytest output. Do NOT call autonomously: only when the learner "
            "explicitly asks (e.g. '跑一下 patch-test' / 'run the test'). Per "
            "house rules you do not execute patch-test before stage 3."
        ),
        "input_schema": {
            "type": "object",
            "required": ["mission"],
            "properties": {
                "mission": {"type": "string"},
            },
        },
    },
]


TOOL_DISPATCH = {
    "navigate_to_source": navigate_to_source,
    "read_file": read_file,
    "search_code": search_code,
    "write_starter": write_starter,
    "run_patch_test": run_patch_test,
}


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"error": "tool input must be an object"}
    try:
        return fn(args)
    except Exception as exc:  # noqa: BLE001 - any tool error becomes a model-visible error
        logger.exception("tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
