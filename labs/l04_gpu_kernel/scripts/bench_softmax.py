from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.gpu.triton_softmax import simulate_softmax_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=int, default=4096)
parser.add_argument("--block", type=int, default=1024)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()
print(
    json.dumps(
        simulate_softmax_kernel(batch=args.batch, seq_len=args.seq, block_size=args.block),
        ensure_ascii=False,
        indent=2,
    )
)
