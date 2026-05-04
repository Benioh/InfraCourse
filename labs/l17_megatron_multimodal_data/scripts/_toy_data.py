from __future__ import annotations

import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "multimodal_toy"
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.jsonl"
SHARDS = DATA / "shards"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_ppm(path: Path, r: int, g: int, b: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"P3\n2 2\n255\n{r} {g} {b} {r} {g} {b} {r} {g} {b} {r} {g} {b}\n"
    path.write_text(body, encoding="ascii")


def make_wav(path: Path, freq: int = 440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        frames = bytearray()
        for i in range(800):
            val = int(12000 * math.sin(2 * math.pi * freq * i / 8000))
            frames.extend(val.to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(frames))
