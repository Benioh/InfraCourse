from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("log", nargs="?", default="rl.log")
args = parser.parse_args()
text = Path(args.log).read_text(encoding="utf-8") if Path(args.log).exists() else ""
print(
    json.dumps(
        {
            "reward_self_test": "reward_self_test=True" in text,
            "lines": len(text.splitlines()),
        },
        ensure_ascii=False,
        indent=2,
    )
)
