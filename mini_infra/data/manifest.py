from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_infra.observability.io import write_json


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    samples = [
        {
            "id": "mm_001",
            "text": "A small image-caption sample",
            "image": "images/mm_001.jpg",
            "audio": None,
        },
        {
            "id": "mm_002",
            "text": "A small audio-text sample",
            "image": None,
            "audio": "audio/mm_002.wav",
        },
    ]
    rows = []
    for sample in samples:
        checks = {}
        for field in ["image", "audio"]:
            value = sample.get(field)
            checks[f"{field}_exists"] = (root / value).exists() if value else None
        rows.append({**sample, **checks})
    payload = {
        "format": "mini_infra_multimodal_manifest_v1",
        "root": str(root),
        "samples": rows,
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny multimodal manifest with schema checks"
    )
    parser.add_argument("--root", default="data/mini_multimodal")
    parser.add_argument("--output", default="runs/mini_infra/data/multimodal_manifest.json")
    args = parser.parse_args()
    print(
        json.dumps(
            build_manifest(Path(args.root), Path(args.output)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
