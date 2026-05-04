from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "wikitext"
RAW_DIR = DATA_DIR / "raw"


def convert_split(split: str) -> int:
    source = RAW_DIR / f"{split}.txt"
    target = DATA_DIR / f"{split}.jsonl"
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for line in source.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    counts = {split: convert_split(split) for split in ("train", "validation", "test")}
    (DATA_DIR / "jsonl_report.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
