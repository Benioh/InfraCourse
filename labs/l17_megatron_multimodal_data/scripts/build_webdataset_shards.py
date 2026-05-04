from __future__ import annotations
import json
import tarfile
from pathlib import Path
from _toy_data import MANIFEST, SHARDS, read_jsonl

SHARDS.mkdir(parents=True, exist_ok=True)
shard = SHARDS / "shard-000000.tar"
rows = read_jsonl(MANIFEST)
with tarfile.open(shard, "w") as tar:
    for row in rows:
        key = row["id"]
        meta = SHARDS / f"{key}.json"
        meta.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        tar.add(meta, arcname=f"{key}.json")
        meta.unlink()
        for kind, file in row["files"].items():
            suffix = Path(file).suffix or f".{kind}"
            tar.add(file, arcname=f"{key}.{kind}{suffix}")
print(shard)
