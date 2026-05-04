from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段 pipeline toy")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch

    torch.manual_seed(1)
    stage0 = torch.nn.Linear(4, 4)
    stage1 = torch.nn.Linear(4, 2)
    outputs = []
    schedule = []
    for microbatch in range(args.microbatches):
        x = torch.randn(2, 4)
        h = torch.relu(stage0(x))
        y = stage1(h)
        outputs.append(y.detach())
        schedule.append(
            {"microbatch": microbatch, "stage0": "forward", "stage1": "forward"}
        )
    bubble_ratio = max(
        0.0, 1 - (args.microbatches * 2) / ((args.microbatches + 2 - 1) * 2)
    )
    payload = {
        "microbatches": args.microbatches,
        "output_batches": len(outputs),
        "bubble_ratio": round(bubble_ratio, 4),
        "schedule": schedule,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
