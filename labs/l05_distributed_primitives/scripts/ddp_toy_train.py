from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def train(steps: int = 30) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch import nn

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    backend = "nccl" if torch.cuda.is_available() and distributed else "gloo"
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    torch.manual_seed(2026 + rank)
    model = nn.Linear(1, 1).to(device)
    if distributed:
        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    x = torch.linspace(-1, 1, 64, device=device).unsqueeze(1)
    y = 2.0 * x + 0.5
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    payload = {
        "rank": rank,
        "world_size": world_size,
        "backend": backend,
        "device": str(device),
        "steps": steps,
        "loss_start": round(losses[0], 6),
        "loss_end": round(losses[-1], 6),
        "loss_decreased": losses[-1] < losses[0],
    }
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="中文 DDP toy training")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = train(args.steps)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output and payload["rank"] == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
