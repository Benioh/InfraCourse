from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def simulated_payload(note: str) -> dict[str, Any]:
    return {
        "backend": "simulated-no-torch",
        "rank": 0,
        "local_rank": 0,
        "world_size": 2,
        "device": "cpu",
        "all_reduce_sum": 3.0,
        "barrier_ok": True,
        "fallback_used": True,
        "note": note,
    }


def run_hello() -> dict[str, Any]:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        return simulated_payload(f"PyTorch 不可用，执行语义模拟：{exc}")

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
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
    else:
        tensor = torch.tensor([1.0], device=device)

    payload = {
        "backend": backend,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "all_reduce_sum": float(tensor.item()),
        "barrier_ok": True,
        "fallback_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if distributed:
        dist.destroy_process_group()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="L01.5 process group hello")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_hello()
    if payload.get("rank", 0) == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
