from __future__ import annotations

from dataclasses import dataclass

from mini_infra.megatron.training.arguments import MiniMegatronArgs


@dataclass
class DistributedState:
    world_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    data_parallel_size: int
    sequence_parallel: bool


def initialize_megatron(args: MiniMegatronArgs) -> DistributedState:
    model_parallel = max(
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size, 1
    )
    world_size = max(model_parallel, 1)
    return DistributedState(
        world_size=world_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        data_parallel_size=max(world_size // model_parallel, 1),
        sequence_parallel=args.sequence_parallel,
    )
