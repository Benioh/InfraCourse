from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_infra.data.toy_data import default_data_path, read_jsonl, sample_text
from mini_infra.observability.io import write_json


def build_indexed_dataset(input_path: Path, output_prefix: Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    offsets = []
    cursor = 0
    with bin_path.open("wb") as bin_handle:
        for row in rows:
            payload = sample_text(row).encode("utf-8") + b"\n"
            offsets.append({"id": row["id"], "offset": cursor, "length": len(payload)})
            bin_handle.write(payload)
            cursor += len(payload)
    write_json(idx_path, {"format": "mini_infra_indexed_text_v1", "records": offsets})
    return {
        "records": len(rows),
        "bin": str(bin_path),
        "idx": str(idx_path),
        "prefix": str(output_prefix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Megatron-style tiny indexed dataset")
    parser.add_argument("--input", default=str(default_data_path()))
    parser.add_argument("--output-prefix", default="runs/mini_infra/data/indexed/toy_text_document")
    args = parser.parse_args()
    payload = build_indexed_dataset(Path(args.input), Path(args.output_prefix))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
