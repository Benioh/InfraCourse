"""Reference solution for L08.8 Patch."""

from __future__ import annotations

import re
from typing import Any

_NUMBER_RE = re.compile(r"-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")


def extract_first_number(text: str) -> float | None:
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    raw = match.group(0)
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def score_exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip().lower() == reference.strip().lower()


def score_first_number_match(prediction: str, reference: str, atol: float = 1e-6) -> bool:
    pred = extract_first_number(prediction)
    ref = extract_first_number(reference)
    if pred is None or ref is None:
        return False
    return abs(pred - ref) <= atol


def format_few_shot_prompt(
    question: str,
    shots: list[tuple[str, str]],
    system: str = "Solve the math problem.",
) -> str:
    parts: list[str] = [system.strip(), ""]
    for q, a in shots:
        parts.append(f"Question: {q.strip()}")
        parts.append(f"Answer: {a.strip()}")
        parts.append("")
    parts.append(f"Question: {question.strip()}")
    parts.append("Answer:")
    return "\n".join(parts)


def run_evaluation(
    items: list[dict],
    client: Any,
    n_shots: int = 8,
    max_tokens: int = 256,
) -> dict:
    if len(items) <= n_shots:
        raise ValueError("need more items than shots")
    shots = [(item["question"], item["answer"]) for item in items[:n_shots]]
    eval_items = items[n_shots:]
    samples = []
    em_correct = 0
    fnm_correct = 0
    for item in eval_items:
        prompt = format_few_shot_prompt(item["question"], shots)
        prediction = client.complete(prompt, max_tokens=max_tokens, stop=["Question:"])
        em = score_exact_match(prediction, item["answer"])
        fnm = score_first_number_match(prediction, item["answer"])
        samples.append(
            {
                "question": item["question"],
                "reference": item["answer"],
                "prediction": prediction,
                "exact_match": em,
                "first_number_match": fnm,
            }
        )
        em_correct += int(em)
        fnm_correct += int(fnm)
    n = len(eval_items)
    return {
        "exact_match": em_correct / max(1, n),
        "first_number_match": fnm_correct / max(1, n),
        "n_total": n,
        "samples": samples,
    }
