from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = ROOT / "data" / "gsm8k_toy" / "train.jsonl"
dst = ROOT / "data" / "gsm8k_toy" / "prompts.jsonl"
rows = []
for line in src.read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    rows.append(
        {
            "prompt": f"请解答数学题，并在最后给出数字答案：{item['question']}",
            "target": item["answer"],
        }
    )
dst.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
print(dst)
