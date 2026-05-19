"""Provider drivers + tool-loop runner for the AI tutor.

`stream_response` is the single entry point. It yields normalized event dicts:

    {"type": "text_delta", "delta": "..."}
    {"type": "tool_use",   "name": "...", "input": {...}}
    {"type": "tool_result","name": "...", "output": {...}}
    {"type": "done"}
    {"type": "error",      "message": "..."}

SDKs are imported lazily so an install without `anthropic` or `openai` doesn't
break the rest of the FastAPI app.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from app.backend.ai_config import ProviderConfig, get_provider
from app.backend.ai_tools import TOOL_SCHEMA, VIBECODING_TOOL_SCHEMA, dispatch
from app.backend.repository import ROOT, patch_payload, source_payload

logger = logging.getLogger(__name__)

MAX_TOOL_LOOPS = 6
MAX_TOKENS = 4096
SELECTION_LIMIT = 8000
TASK_MD_LIMIT = 6000
SOURCE_PREVIEW_LIMIT = 12000
VISIBLE_TEXT_LIMIT = 4000
PAGE_SUMMARY_LIMIT = 1500
VIBE_DOC_LIMIT = 8000


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


_TUTOR_PREAMBLE = [
    "You are an in-app AI tutor for the Infra Quest course (multimodal LLM "
    "infrastructure training: Megatron, vLLM, SGLang, verl, SLiME, etc.).",
    "Help the learner understand the framework source code they are reading.",
    "When a question targets specific code, prefer reading or searching the "
    "real files via the provided tools instead of guessing.",
    "When you can point the learner at a specific spot, emit "
    "`navigate_to_source` so the in-app viewer jumps there.",
    "Answer in the language the learner uses (中文 or English). Be concise; "
    "show short code excerpts only when they make the answer clearer.",
]

_CODER_PREAMBLE_FALLBACK = [
    "You are pair-programming inside an Infra Quest lab in Vibe Coding mode.",
    "Strict house rules:",
    "1) Before writing ANY code, confirm you have: lab id, patch contract from "
    "task.md (input/output/shape/dtype/invariants), the source map paths, and "
    "a one-line guess from the learner. If anything is missing, ask first.",
    "2) Split each lab into ≥3 stages (skeleton → core algo → boundary cases). "
    "After each stage, ask 1-2 checkpoint questions (Contract / Mapping / "
    "Counterfactual) and WAIT for the learner before continuing.",
    "3) Never write a complete solution in one shot. Never copy production code "
    "from github_repo/ into mini_infra/.",
    "4) When the learner triggers a write, call `write_starter` with stage 1, "
    "2, or 3. After write_starter, do NOT call run_patch_test yourself unless "
    "the learner asks. They run it; you analyze.",
    "5) On failing patch tests, walk a hint ladder L0→L4 (test name & "
    "expected/actual → region → semantic explanation → ≤3-line minimal fix → "
    "full patch only if the learner says give me the answer).",
    "Reply in the language the learner uses (中文 or English).",
]


def _load_vibe_doc() -> str | None:
    path = ROOT / "docs" / "AI_tutor" / "AI_vibe_coding协作约束.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text[:VIBE_DOC_LIMIT]


def build_system_prompt(
    context: dict[str, Any] | None,
    mode: str = "tutor",
) -> str:
    parts: list[str]
    if mode == "coder":
        vibe_doc = _load_vibe_doc()
        if vibe_doc:
            parts = [
                "You are pair-programming inside an Infra Quest lab in Vibe "
                "Coding mode. Follow the project's协作约束 below verbatim:",
                "\n## docs/AI_tutor/AI_vibe_coding协作约束.md\n" + vibe_doc,
                "\nWhen the learner has confirmed contract/source-map/guess and "
                "you are ready to commit a stage, call `write_starter` with the "
                "stage number; do not autonomously call `run_patch_test`. Reply "
                "in the language the learner uses.",
            ]
        else:
            parts = list(_CODER_PREAMBLE_FALLBACK)
    else:
        parts = list(_TUTOR_PREAMBLE)

    if not context:
        return "\n".join(parts)

    page_kind = context.get("page_kind")
    if isinstance(page_kind, str) and page_kind:
        parts.append(f"\n## Page\nThe learner is currently on a `{page_kind}` page.")

    page_summary = context.get("page_summary")
    if isinstance(page_summary, str) and page_summary.strip():
        parts.append(f"\n## Page summary\n{page_summary[:PAGE_SUMMARY_LIMIT]}")

    selection = context.get("selection")
    if isinstance(selection, str) and selection.strip():
        text = selection[:SELECTION_LIMIT]
        parts.append(f"\n## Learner selection\n```\n{text}\n```")

    source_path = context.get("source_path")
    lines = context.get("lines")
    if isinstance(source_path, str) and source_path:
        header = f"\n## Source context\nFile: `{source_path}`"
        if isinstance(lines, list) and len(lines) == 2:
            header += f"  Lines: L{lines[0]}-L{lines[1]}"
        parts.append(header)
        try:
            payload = source_payload(source_path)
        except (PermissionError, FileNotFoundError, IsADirectoryError):
            payload = None
        if payload and not payload.get("truncated"):
            content_lines = (payload.get("content") or "").splitlines()
            if isinstance(lines, list) and len(lines) == 2:
                a = max(1, lines[0] - 50)
                b = min(len(content_lines), lines[1] + 50)
            else:
                a, b = 1, min(len(content_lines), 200)
            if b >= a:
                snippet = "\n".join(
                    f"{i:5d} | {ln}"
                    for i, ln in enumerate(content_lines[a - 1 : b], start=a)
                )
                snippet = snippet[:SOURCE_PREVIEW_LIMIT]
                parts.append(f"\nFile preview L{a}-L{b}:\n```\n{snippet}\n```")

    notebook_path = context.get("notebook_path")
    if isinstance(notebook_path, str) and notebook_path:
        parts.append(f"\n## Notebook\nThe learner is reading `{notebook_path}`.")

    ticket_id = context.get("ticket_id")
    if isinstance(ticket_id, str) and ticket_id:
        parts.append(f"\n## Ticket\nThe learner is on debug ticket `{ticket_id}`.")

    mission_id = context.get("mission_id")
    if isinstance(mission_id, str) and mission_id:
        try:
            mission = patch_payload(mission_id)
        except FileNotFoundError:
            mission = None
        if mission:
            task_md = mission.get("task_md_content") or ""
            if task_md:
                parts.append(
                    f"\n## Mission `{mission_id}` task.md\n{task_md[:TASK_MD_LIMIT]}"
                )

    visible_text = context.get("visible_text")
    if isinstance(visible_text, str) and visible_text.strip():
        parts.append(
            "\n## What the learner is currently looking at (page text)\n"
            + visible_text[:VISIBLE_TEXT_LIMIT]
        )

    return "\n".join(parts)


def _tools_for_mode(mode: str) -> list[dict[str, Any]]:
    return VIBECODING_TOOL_SCHEMA if mode == "coder" else TOOL_SCHEMA


# ---------------------------------------------------------------------------
# Per-provider tool-shape helpers
# ---------------------------------------------------------------------------


def _anthropic_tools(mode: str) -> list[dict[str, Any]]:
    return _tools_for_mode(mode)


def _openai_tools(mode: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        for t in _tools_for_mode(mode)
    ]


# ---------------------------------------------------------------------------
# Claude (Anthropic Messages API) streaming
# ---------------------------------------------------------------------------


async def _stream_claude(
    cfg: ProviderConfig,
    system: str,
    history: list[dict[str, Any]],
    user_message: str,
    mode: str = "tutor",
) -> AsyncIterator[dict[str, Any]]:
    import anthropic

    client = anthropic.AsyncAnthropic(base_url=cfg.base_url, api_key=cfg.api_key)
    messages: list[dict[str, Any]] = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in history
        if m.get("role") in {"user", "assistant"}
    ]
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_LOOPS):
        async with client.messages.stream(
            model=cfg.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=_anthropic_tools(mode),
        ) as stream:
            async for event in stream:
                if (
                    getattr(event, "type", None) == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"
                ):
                    yield {"type": "text_delta", "delta": event.delta.text}
            final = await stream.get_final_message()

        stop_reason = getattr(final, "stop_reason", None)
        if stop_reason != "tool_use":
            yield {"type": "done"}
            return

        # Anthropic requires the assistant turn (text + tool_use blocks) to be
        # appended verbatim before the tool_result user turn.
        assistant_blocks = [block.model_dump() for block in final.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_results: list[dict[str, Any]] = []
        for block in final.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_name = block.name
            tool_input = block.input or {}
            yield {"type": "tool_use", "name": tool_name, "input": tool_input}
            output = await asyncio.to_thread(dispatch, tool_name, tool_input)
            yield {"type": "tool_result", "name": tool_name, "output": output}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
        if not tool_results:
            yield {"type": "done"}
            return
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": "tool loop exceeded MAX_TOOL_LOOPS"}


# ---------------------------------------------------------------------------
# Codex (OpenAI Responses API) streaming
# ---------------------------------------------------------------------------


async def _stream_codex(
    cfg: ProviderConfig,
    system: str,
    history: list[dict[str, Any]],
    user_message: str,
    mode: str = "tutor",
) -> AsyncIterator[dict[str, Any]]:
    import openai

    client = openai.AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    input_items: list[dict[str, Any]] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            input_items.append({"role": role, "content": content})
    input_items.append({"role": "user", "content": user_message})

    extras: dict[str, Any] = {}
    reasoning_effort = cfg.extras.get("reasoning_effort")
    if reasoning_effort:
        extras["reasoning"] = {"effort": reasoning_effort}

    for _ in range(MAX_TOOL_LOOPS):
        async with client.responses.stream(
            model=cfg.model,
            input=input_items,
            tools=_openai_tools(mode),
            instructions=system,
            **extras,
        ) as stream:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield {"type": "text_delta", "delta": delta}
            final = await stream.get_final_response()

        function_calls = [
            item for item in final.output if getattr(item, "type", None) == "function_call"
        ]
        if not function_calls:
            yield {"type": "done"}
            return

        # Pass back ALL output items (text, reasoning, function_calls) so the
        # next turn has continuity, then append function_call_output items.
        for item in final.output:
            input_items.append(item.model_dump())

        for fc in function_calls:
            try:
                args = json.loads(fc.arguments) if fc.arguments else {}
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_use", "name": fc.name, "input": args}
            output = await asyncio.to_thread(dispatch, fc.name, args)
            yield {"type": "tool_result", "name": fc.name, "output": output}
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                }
            )

    yield {"type": "error", "message": "tool loop exceeded MAX_TOOL_LOOPS"}


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


async def stream_response(
    provider_name: str,
    system: str,
    history: list[dict[str, Any]],
    user_message: str,
    mode: str = "tutor",
) -> AsyncIterator[dict[str, Any]]:
    try:
        cfg = get_provider(provider_name)
    except LookupError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    try:
        if provider_name == "claude":
            async for event in _stream_claude(cfg, system, history, user_message, mode):
                yield event
        elif provider_name == "codex":
            async for event in _stream_codex(cfg, system, history, user_message, mode):
                yield event
        else:
            yield {"type": "error", "message": f"unknown provider: {provider_name}"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any SDK error to the client
        logger.exception("stream_response failed")
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
