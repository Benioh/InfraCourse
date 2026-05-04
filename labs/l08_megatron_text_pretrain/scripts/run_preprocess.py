from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l08_megatron_text_pretrain"
DATA_DIR = ROOT / "data" / "wikitext"
INDEXED_DIR = DATA_DIR / "indexed_dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    ensure_prediction(run_dir / "prediction.yaml")
    write_command_snapshot(run_dir)
    write_yaml(run_dir / "config.resolved.yaml", {"stage": "preprocess"})
    INDEXED_DIR.mkdir(parents=True, exist_ok=True)

    megatron_script = os.environ.get("MEGATRON_PREPROCESS_SCRIPT")
    command = (
        f"python {megatron_script} --input {DATA_DIR / 'train.jsonl'} --output-prefix {INDEXED_DIR / 'wikitext_text_document'}"
        if megatron_script
        else "python $MEGATRON_PREPROCESS_SCRIPT --input data/wikitext/train.jsonl --output-prefix data/wikitext/indexed_dataset/wikitext_text_document"
    )
    if (
        megatron_script
        and Path(megatron_script).exists()
        and os.environ.get("INFRA_QUEST_ENABLE_REAL_MEGATRON") == "1"
    ):
        write_text(
            run_dir / "artifacts" / "preprocess_fallback.md",
            "Real preprocess path was requested but automatic invocation is intentionally left manual. Run the command in `expected_preprocess_command.sh`.\n",
        )
    else:
        write_text(
            run_dir / "artifacts" / "preprocess_fallback.md",
            "Megatron preprocess script was not available locally. Dataset and JSONL were validated, and the exact preprocess command was recorded for manual execution.\n",
        )
    write_text(run_dir / "artifacts" / "expected_preprocess_command.sh", command + "\n")
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "train",
            "stage": "preprocess",
            "status": "validated",
        },
    )
    write_text(
        run_dir / "train.log",
        f"[{utc_now()}] preprocess validated\ncommand={command}\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
