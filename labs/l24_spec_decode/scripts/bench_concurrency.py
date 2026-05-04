from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.vllm.spec_decode.draft_runner import spec_decode_summary

parser = argparse.ArgumentParser()
parser.add_argument("--servers", default="8001,8002,8003")
parser.add_argument("--batches", default="1,16,64")
args = parser.parse_args()
rows = []
for batch in [int(x) for x in args.batches.split(",") if x]:
    rows.append(spec_decode_summary("ngram", concurrency=batch))
    rows.append(spec_decode_summary("draft", concurrency=batch))
print(json.dumps({"servers": args.servers.split(","), "rows": rows}, ensure_ascii=False, indent=2))
