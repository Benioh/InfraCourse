from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any

NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


def extract_final_number(text: str) -> float | None:
    match = re.search(r"(?:####|final answer\s*:?)\s*(%s)" % NUMBER, text, flags=re.I)
    values = [match.group(1)] if match else re.findall(NUMBER, text)
    if not values:
        return None
    value = float(values[-1].replace(",", ""))
    return int(value) if value.is_integer() else value


def score(prediction: str, target: str) -> dict[str, Any]:
    prediction_value = extract_final_number(prediction)
    target_value = extract_final_number(target)
    reward = float(
        prediction_value is not None
        and target_value is not None
        and math.isclose(float(prediction_value), float(target_value), abs_tol=1e-6)
    )
    return {
        "prediction_final": prediction_value,
        "target_final": target_value,
        "reward": reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra reward parser")
    parser.add_argument("--prediction", default="Final answer: 7")
    parser.add_argument("--target", default="#### 7")
    args = parser.parse_args()
    print(json.dumps(score(args.prediction, args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
