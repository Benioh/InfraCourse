from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch
import torch.distributed as dist


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    can_use_cuda = gpu_count >= world_size and gpu_count > 0
    backend = "nccl" if can_use_cuda else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if can_use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    payload = {
        "timestamp": utc_now(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": backend,
        "device": device,
    }
    print(json.dumps(payload))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
