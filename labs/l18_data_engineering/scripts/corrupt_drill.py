from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_infra.data.shard_resume import recovery_plan

parser = argparse.ArgumentParser()
parser.add_argument("--shard", default="data/shards/0001.tar")
args = parser.parse_args()
shards = ["data/shards/0000.tar", args.shard, "data/shards/0002.tar"]
print(json.dumps(recovery_plan(shards, corrupt_shards={args.shard}), ensure_ascii=False, indent=2))
