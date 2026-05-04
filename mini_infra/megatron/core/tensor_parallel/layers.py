from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorParallelShard:
    rank: int
    start: int
    end: int


class ColumnParallelLinear:
    """Minimal ColumnParallelLinear: split output columns exactly like Megatron's concept."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tensor_model_parallel_size: int = 1,
        gather_output: bool = True,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_model_parallel_size = max(tensor_model_parallel_size, 1)
        self.gather_output = gather_output
        shard = output_size // self.tensor_model_parallel_size
        self.shards = [
            TensorParallelShard(
                rank,
                rank * shard,
                (
                    output_size
                    if rank == self.tensor_model_parallel_size - 1
                    else (rank + 1) * shard
                ),
            )
            for rank in range(self.tensor_model_parallel_size)
        ]

    def partition(self) -> list[TensorParallelShard]:
        return self.shards

    def communication(self) -> str:
        return (
            "all_gather_output"
            if self.gather_output and self.tensor_model_parallel_size > 1
            else "local_output"
        )


class RowParallelLinear:
    """Minimal RowParallelLinear: split input rows and reduce partial outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tensor_model_parallel_size: int = 1,
        input_is_parallel: bool = False,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.tensor_model_parallel_size = max(tensor_model_parallel_size, 1)
        self.input_is_parallel = input_is_parallel
        shard = input_size // self.tensor_model_parallel_size
        self.shards = [
            TensorParallelShard(
                rank,
                rank * shard,
                (
                    input_size
                    if rank == self.tensor_model_parallel_size - 1
                    else (rank + 1) * shard
                ),
            )
            for rank in range(self.tensor_model_parallel_size)
        ]

    def partition(self) -> list[TensorParallelShard]:
        return self.shards

    def communication(self) -> str:
        return (
            "reduce_scatter_or_all_reduce"
            if self.tensor_model_parallel_size > 1
            else "local_output"
        )
