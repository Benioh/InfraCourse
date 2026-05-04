from __future__ import annotations
from _toy_data import RAW, make_text, make_wav

for i, freq in enumerate([300, 420], 1):
    make_wav(RAW / "librispeech" / f"audio_{i}.wav", freq)
    make_text(RAW / "librispeech" / f"audio_{i}.txt", f"这是第 {i} 条教学语音转写。")
print("已生成 LibriSpeech 练习子集")
