from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerShard:
    rank: int
    parameter_start: int
    parameter_end: int
    owns_optimizer_state: bool = True


class DistributedOptimizer:
    """Tiny optimizer-state sharding model matching Megatron distributed optimizer concepts."""

    def __init__(self, parameter_count: int, data_parallel_size: int) -> None:
        self.parameter_count = parameter_count
        self.data_parallel_size = max(data_parallel_size, 1)
        self.param_groups = [{"lr": 1.0e-4}]
        self.step_count = 0
        shard = parameter_count // self.data_parallel_size
        self.shards = [
            OptimizerShard(
                rank,
                rank * shard,
                (
                    parameter_count
                    if rank == self.data_parallel_size - 1
                    else (rank + 1) * shard
                ),
            )
            for rank in range(self.data_parallel_size)
        ]

    def state_memory_ratio(self) -> float:
        return round(1 / self.data_parallel_size, 4)

    def zero_grad(self) -> None:
        return None

    def step(self) -> dict[str, float | bool]:
        self.step_count += 1
        return {"success": True, "grad_norm": round(1.0 / self.step_count, 6)}

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "distributed": self.data_parallel_size > 1,
            "shards": self.checkpoint_layout(),
        }

    def checkpoint_layout(self) -> list[dict[str, int | bool]]:
        return [shard.__dict__ for shard in self.shards]
