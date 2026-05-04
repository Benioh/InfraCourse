from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_infra.observability.io import ROOT, write_text

SAMPLES = [
    {
        "id": "math_001",
        "prompt": "Alice has 3 apples and buys 4 more. Answer:",
        "answer": "7",
    },
    {
        "id": "math_002",
        "prompt": "Tom reads 2 pages per minute for 8 minutes. Answer:",
        "answer": "16",
    },
    {
        "id": "math_003",
        "prompt": "There are 9 birds and 3 fly away. Answer:",
        "answer": "6",
    },
    {
        "id": "math_004",
        "prompt": "A box has 5 red balls and 6 blue balls. Answer:",
        "answer": "11",
    },
    {
        "id": "math_005",
        "prompt": "Mia saves 12 dollars and spends 5. Answer:",
        "answer": "7",
    },
]


def sample_text(row: dict[str, str]) -> str:
    return f"Question: {row['prompt']} Final answer: {row['answer']}"


def write_jsonl(path: Path, rows: list[dict[str, Any]] | None = None) -> None:
    payload = rows or SAMPLES
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        write_jsonl(path)
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def default_data_path() -> Path:
    return ROOT / "mini_infra" / "data" / "toy_math.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write MiniInfra toy math dataset")
    parser.add_argument("--output", default=str(default_data_path()))
    args = parser.parse_args()
    output = Path(args.output)
    write_jsonl(output)
    print(output)


if __name__ == "__main__":
    main()
