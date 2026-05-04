from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import ensure_dir

DATA_DIR = ROOT / "data" / "wikitext"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    ensure_dir(RAW_DIR)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    summary = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows = [row["text"] for row in dataset[split]]
        summary["splits"][split] = len(rows)
        (RAW_DIR / f"{split}.txt").write_text("\n".join(rows), encoding="utf-8")
    sample_check = []
    for split in ("train", "validation", "test"):
        non_empty = [
            line
            for line in (RAW_DIR / f"{split}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        sample_check.extend(non_empty[:2])
    summary["validated_samples"] = min(len(sample_check), 5)
    summary["ready"] = True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "download_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "DATASET_READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
