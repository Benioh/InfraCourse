from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_infra.observability.io import write_json, write_text


def convert_to_hf(training_checkpoint: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = training_checkpoint / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    write_json(
        output_dir / "config.json",
        {
            "model_type": "mini_infra_tiny_lm",
            "source_manifest": str(manifest_path),
            "model_config": manifest.get("model_config", {}),
        },
    )
    write_json(
        output_dir / "tokenizer.json",
        {
            "type": "byte_level_minimal",
            "vocab_size": manifest.get("model_config", {}).get("vocab_size", 128),
        },
    )
    write_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {"format": "minimal_hf_index"},
            "weight_map": {"model.weight": "model-00001-of-00001.safetensors"},
        },
    )
    write_text(
        output_dir / "model-00001-of-00001.safetensors",
        "minimal tensor payload; replace with real converted weights for serving\n",
    )
    return {"hf_dir": str(output_dir), "source_checkpoint": str(training_checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MiniInfra training checkpoint to minimal HF serving layout"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_to_hf(Path(args.checkpoint), Path(args.output)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
