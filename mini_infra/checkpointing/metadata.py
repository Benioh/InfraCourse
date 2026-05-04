from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_infra.observability.io import utc_now, write_json


def checkpoint_manifest(
    checkpoint_dir: Path,
    checkpoint_type: str,
    model_config: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = (
        sorted(
            str(path.relative_to(checkpoint_dir))
            for path in checkpoint_dir.rglob("*")
            if path.is_file()
        )
        if checkpoint_dir.exists()
        else []
    )
    return {
        "format": "mini_infra_checkpoint_manifest_v1",
        "checkpoint_type": checkpoint_type,
        "created_at": utc_now(),
        "model_config": model_config,
        "training_state": training_state or {},
        "files": files,
        "compatibility_notes": [
            "serving checkpoint needs model weights plus tokenizer/config metadata",
            "distributed checkpoint also needs parallelism and optimizer state metadata",
        ],
    }


def write_manifest(
    checkpoint_dir: Path,
    checkpoint_type: str,
    model_config: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_manifest(
        checkpoint_dir, checkpoint_type, model_config, training_state
    )
    write_json(checkpoint_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write MiniInfra checkpoint manifest")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-type", default="training")
    args = parser.parse_args()
    payload = write_manifest(Path(args.checkpoint_dir), args.checkpoint_type, {})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
