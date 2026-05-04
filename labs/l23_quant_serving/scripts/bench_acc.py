from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.vllm.run_engine import quant_payload

parser = argparse.ArgumentParser()
parser.add_argument("--eval", default="gsm8k")
parser.add_argument("--servers", default="8001,8002,8003,8004")
args = parser.parse_args()
rows = [quant_payload(kind) for kind in ["none", "awq", "fp8", "kvint8"]]
print(
    json.dumps(
        {"eval": args.eval, "servers": args.servers.split(","), "rows": rows},
        ensure_ascii=False,
        indent=2,
    )
)
