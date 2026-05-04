from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MiniMegatronArgs:
    train_samples: int = 8
    seq_length: int = 32
    micro_batch_size: int = 2
    global_batch_size: int = 4
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    sequence_parallel: bool = False
    use_distributed_optimizer: bool = False
    save: str = "runs/mini_infra/megatron/checkpoints"
    data_path: str = "mini_infra/data/toy_math.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
