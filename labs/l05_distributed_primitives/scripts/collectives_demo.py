from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def run_collectives() -> dict[str, Any]:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        values = [1.0, 2.0]
        return {
            "backend": "simulated-no-torch",
            "rank": 0,
            "world_size": len(values),
            "all_reduce_sum": sum(values),
            "all_gather": values,
            "broadcast_value": values[0],
            "all_reduce_ms": 0.0,
            "note": f"PyTorch 不可用，执行纯 Python 语义模拟：{exc}",
        }

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

    tensor = torch.tensor([float(rank + 1)], device=device)
    start = time.perf_counter()
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        tensor = torch.tensor([sum(range(1, world_size + 1)) or 1.0], device=device)
    all_reduce_ms = (time.perf_counter() - start) * 1000

    gather_slots = [torch.zeros_like(tensor) for _ in range(world_size)]
    if distributed:
        dist.all_gather(gather_slots, torch.tensor([float(rank + 1)], device=device))
    else:
        gather_slots = [torch.tensor([1.0], device=device)]

    broadcast = torch.tensor([float(rank + 10)], device=device)
    if distributed:
        if rank == 0:
            broadcast.fill_(123.0)
        dist.broadcast(broadcast, src=0)
    else:
        broadcast.fill_(123.0)

    payload = {
        "backend": backend,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "all_reduce_sum": float(tensor.item()),
        "all_gather": [float(x.item()) for x in gather_slots],
        "broadcast_value": float(broadcast.item()),
        "all_reduce_ms": round(all_reduce_ms, 4),
    }
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="中文 collective 教学脚本")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = run_collectives()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output and payload.get("rank", 0) == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
