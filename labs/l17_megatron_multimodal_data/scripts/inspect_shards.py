from __future__ import annotations
import argparse
import json
import tarfile
from pathlib import Path
from _toy_data import SHARDS

parser = argparse.ArgumentParser()
parser.add_argument("--output")
args = parser.parse_args()
items = []
for shard in sorted(SHARDS.glob("*.tar")):
    with tarfile.open(shard) as tar:
        names = tar.getnames()
        items.append(
            {
                "shard": str(shard),
                "members": len(names),
                "sample_keys": sorted({n.split(".")[0] for n in names}),
            }
        )
payload = {"shard_count": len(items), "shards": items}
print(json.dumps(payload, indent=2, ensure_ascii=False))
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
