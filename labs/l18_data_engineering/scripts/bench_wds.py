from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.data.wds_pipeline import simulate_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--prefetch", type=int, default=4)
args = parser.parse_args()
print(
    json.dumps(
        simulate_pipeline(workers=args.workers, prefetch=args.prefetch),
        ensure_ascii=False,
        indent=2,
    )
)
