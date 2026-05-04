"""L08.8 Patch · 最小评测 harness。"""

from __future__ import annotations

import re
from typing import Any


def extract_first_number(text: str) -> float | None:
    """Extract the first signed decimal number, ignoring $/% and thousands ','."""
    # TODO(student): regex like r"-?\$?\d{1,3}(,\d{3})*(\.\d+)?%?"
    # TODO(student): strip $ , %; convert to float
    raise NotImplementedError("L08.8: implement extract_first_number")


def score_exact_match(prediction: str, reference: str) -> bool:
    # TODO(student): strip + lower + compare
    raise NotImplementedError


def score_first_number_match(prediction: str, reference: str, atol: float = 1e-6) -> bool:
    # TODO(student): extract numbers from both; if either None -> False; else abs(a-b) <= atol
    raise NotImplementedError


def format_few_shot_prompt(
    question: str,
    shots: list[tuple[str, str]],
    system: str = "Solve the math problem.",
) -> str:
    # TODO(student): start with system; for each (q, a) shot: "Question: ...\nAnswer: ...\n"
    # TODO(student): then add the real question with "Question: ... \nAnswer:"
    raise NotImplementedError


def run_evaluation(
    items: list[dict],
    client: Any,
    n_shots: int = 8,
    max_tokens: int = 256,
) -> dict:
    # TODO(student): take items[:n_shots] as shots
    # TODO(student): for each remaining item, build prompt, call client.complete(prompt, max_tokens=...)
    # TODO(student): score with both metrics; collect samples and aggregate
    # TODO(student): return {"exact_match", "first_number_match", "n_total", "samples"}
    raise NotImplementedError
