from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Column parallel linear 数值校验")
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch

    torch.manual_seed(0)
    x = torch.randn(3, 4)
    weight = torch.randn(4, 8)
    full = x @ weight
    shards = torch.chunk(weight, 2, dim=1)
    gathered = torch.cat([x @ shard for shard in shards], dim=1)
    max_error = float((full - gathered).abs().max())
    payload = {
        "tp_size": 2,
        "input_shape": list(x.shape),
        "output_shape": list(full.shape),
        "max_error": max_error,
        "passed": max_error < 1e-6,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
