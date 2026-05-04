from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default="metrics.jsonl")
args = parser.parse_args()
p = Path(args.path)
rows = (
    [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if p.exists()
    else []
)
print(
    json.dumps(
        {
            "rows": len(rows),
            "configs": sorted({r.get("config") for r in rows if r.get("config")}),
        },
        ensure_ascii=False,
        indent=2,
    )
)
