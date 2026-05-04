from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from _toy_data import MANIFEST, read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--output")
args = parser.parse_args()
allowed = {"image_text_caption", "audio_text_asr", "audio_label"}
rows = read_jsonl(MANIFEST)
missing = []
for row in rows:
    if row.get("task") not in allowed:
        missing.append(f"bad_task:{row.get('id')}")
    for file in row.get("files", {}).values():
        if not Path(file).exists():
            missing.append(file)
payload = {
    "manifest": str(MANIFEST),
    "rows": len(rows),
    "missing_files": missing,
    "validation_passed": not missing,
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
if missing:
    sys.exit(1)
