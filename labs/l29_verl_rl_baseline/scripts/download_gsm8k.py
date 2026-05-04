from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
out = ROOT / "data" / "gsm8k_toy" / "train.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
rows = [
    {"question": "小明有 40 个苹果，又买了 2 个，一共有多少？", "answer": "#### 42"},
    {"question": "3 袋糖每袋 5 个，一共多少？", "answer": "#### 15"},
    {"question": "一本书 12 元，买 2 本多少钱？", "answer": "#### 24"},
    {"question": "10 减 7 等于几？", "answer": "#### 3"},
]
out.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
print(out)
