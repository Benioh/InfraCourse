from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.megatron.core.context_parallel.ring_attention import ring_attention_plan

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=int, default=4096)
parser.add_argument("--cp", type=int, default=1)
args = parser.parse_args()
print(
    json.dumps(ring_attention_plan(seq_len=args.seq, cp_size=args.cp), ensure_ascii=False, indent=2)
)
