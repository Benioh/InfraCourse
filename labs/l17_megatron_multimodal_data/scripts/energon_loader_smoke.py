from __future__ import annotations
import argparse
import json
import tarfile
from pathlib import Path
from _toy_data import SHARDS

parser = argparse.ArgumentParser()
parser.add_argument("--output")
args = parser.parse_args()
samples = []
for shard in sorted(SHARDS.glob("*.tar")):
    with tarfile.open(shard) as tar:
        keys = sorted({name.split(".")[0] for name in tar.getnames()})
        samples.extend(keys)
payload = {
    "loader": "webdataset-compatible-toy",
    "sample_count": len(samples),
    "first_samples": samples[:5],
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
