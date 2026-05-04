from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


def normalize_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def extract_final_number(text: str) -> float | None:
    final_answer = re.search(
        r"(?:####|final answer\s*:?|答案\s*(?:是|:)?)\s*(%s)" % NUMBER, text, flags=re.I
    )
    if final_answer:
        return normalize_number(final_answer.group(1))
    nums = re.findall(NUMBER, text)
    return normalize_number(nums[-1]) if nums else None


def reward(prediction: str, target: str, tolerance: float = 1e-6) -> float:
    pred = extract_final_number(prediction)
    gold = extract_final_number(target)
    if pred is None or gold is None:
        return 0.0
    return float(math.isclose(float(pred), float(gold), abs_tol=tolerance))


def explain_reward(prediction: str, target: str) -> dict[str, Any]:
    pred = extract_final_number(prediction)
    gold = extract_final_number(target)
    value = reward(prediction, target)
    return {
        "prediction_final": pred,
        "target_final": gold,
        "reward": value,
        "failure_reason": (
            None
            if value
            else (
                "missing_number" if pred is None or gold is None else "answer_mismatch"
            )
        ),
    }


def default_cases() -> list[tuple[str, str, float]]:
    return [
        ("答案是 42", "#### 42", 1.0),
        ("41", "#### 42", 0.0),
        ("先算 40，再加 2，所以 42。", "#### 42", 1.0),
        ("Final answer: 1,024", "#### 1024", 1.0),
        ("没有给出最终答案", "#### 7", 0.0),
    ]


def load_cases(path: Path | None) -> list[tuple[str, str, float]]:
    if path is None:
        return default_cases()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(row["prediction"], row["target"], float(row["expected"])) for row in rows]


def self_test(cases_path: Path | None = None) -> dict[str, Any]:
    rows = []
    for prediction, target, expected in load_cases(cases_path):
        observed = reward(prediction, target)
        rows.append(
            {
                "prediction": prediction,
                "target": target,
                "reward": observed,
                "expected": expected,
                "passed": observed == expected,
                **explain_reward(prediction, target),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSM8K reward parser with failure explanations"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cases-file")
    parser.add_argument("--prediction")
    parser.add_argument("--target")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        payload: Any = self_test(Path(args.cases_file) if args.cases_file else None)
    elif args.explain:
        payload = explain_reward(args.prediction or "", args.target or "")
    else:
        payload = reward(args.prediction or "", args.target or "")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(payload, ensure_ascii=False, indent=2)
        if not isinstance(payload, float)
        else payload
    )


if __name__ == "__main__":
    main()
