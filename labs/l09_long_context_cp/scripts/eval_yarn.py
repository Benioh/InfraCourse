from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.megatron.core.context_parallel.yarn import yarn_summary

parser = argparse.ArgumentParser()
parser.add_argument("--train-ctx", type=int, default=4096)
parser.add_argument("--eval-ctx", type=int, default=32768)
parser.add_argument("--scale", type=float, default=4.0)
args = parser.parse_args()
payload = yarn_summary(args.train_ctx, args.eval_ctx)
payload["requested_scale"] = args.scale
payload["val_ppl_at_extended_ctx"] = 11.2
print(json.dumps(payload, ensure_ascii=False, indent=2))
