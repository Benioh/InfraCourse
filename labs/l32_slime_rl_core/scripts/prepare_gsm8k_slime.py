from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = ROOT / "data" / "gsm8k_toy" / "prompts.jsonl"
if not src.exists():
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT / "labs" / "l29_verl_rl_baseline" / "scripts" / "download_gsm8k.py"
            ),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "labs"
                / "l29_verl_rl_baseline"
                / "scripts"
                / "prepare_gsm8k_prompts.py"
            ),
        ],
        check=True,
    )
dst = ROOT / "data" / "gsm8k_toy" / "slime_prompts.jsonl"
rows = [
    json.loads(line) | {"source": "slime"}
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dst.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
print(dst)
