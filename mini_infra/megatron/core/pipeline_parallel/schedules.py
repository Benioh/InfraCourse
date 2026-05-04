from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineEvent:
    time: int
    stage: int
    microbatch: int
    op: str


def get_forward_backward_func(pipeline_model_parallel_size: int):
    return (
        forward_backward_pipelining_without_interleaving
        if pipeline_model_parallel_size > 1
        else forward_backward_no_pipelining
    )


def forward_backward_no_pipelining(num_microbatches: int) -> list[PipelineEvent]:
    return [
        PipelineEvent(
            time=microbatch, stage=0, microbatch=microbatch, op="forward_backward"
        )
        for microbatch in range(num_microbatches)
    ]


def forward_backward_pipelining_without_interleaving(
    num_microbatches: int, stages: int = 2
) -> list[PipelineEvent]:
    events: list[PipelineEvent] = []
    for microbatch in range(num_microbatches):
        for stage in range(stages):
            events.append(
                PipelineEvent(
                    time=microbatch + stage,
                    stage=stage,
                    microbatch=microbatch,
                    op="forward",
                )
            )
    for microbatch in reversed(range(num_microbatches)):
        for stage in reversed(range(stages)):
            events.append(
                PipelineEvent(
                    time=num_microbatches
                    + (num_microbatches - 1 - microbatch)
                    + (stages - 1 - stage),
                    stage=stage,
                    microbatch=microbatch,
                    op="backward",
                )
            )
    return events


def bubble_ratio(num_microbatches: int, stages: int) -> float:
    return round((stages - 1) / max(num_microbatches + stages - 1, 1), 4)
