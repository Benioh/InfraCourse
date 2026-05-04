from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.megatron.core.transformer.moe.experts import moe_summary

parser = argparse.ArgumentParser()
parser.add_argument("--experts", type=int, default=8)
parser.add_argument("--topk", type=int, default=2)
parser.add_argument("--ep", type=int, default=2)
parser.add_argument("--tp", type=int, default=1)
args = parser.parse_args()
payload = moe_summary(num_experts=args.experts, top_k=args.topk, ep_size=args.ep)
payload["tp"] = args.tp
print(json.dumps(payload, ensure_ascii=False, indent=2))
