from __future__ import annotations
from _toy_data import RAW, make_text, make_wav

for i, label in enumerate(["dog", "rain"], 1):
    make_wav(RAW / "esc50" / f"sound_{i}.wav", 500 + i * 80)
    make_text(RAW / "esc50" / f"sound_{i}.label", label)
print("已生成 ESC50 练习子集")
