"""Reference solution for L15 Patch."""

from __future__ import annotations


def make_1f1b_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple[str, int]]]:
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if num_microbatches < num_stages:
        raise ValueError(
            "num_microbatches must be >= num_stages for the 1F1B fill-up to work"
        )
    schedule: list[list[tuple[str, int]]] = []
    for stage in range(num_stages):
        warmup = num_stages - stage - 1
        timeline: list[tuple[str, int]] = []
        next_forward = 0
        next_backward = 0
        for _ in range(warmup):
            timeline.append(("F", next_forward))
            next_forward += 1
        steady_count = num_microbatches - warmup
        for _ in range(steady_count):
            timeline.append(("F", next_forward))
            next_forward += 1
            timeline.append(("B", next_backward))
            next_backward += 1
        for _ in range(warmup):
            timeline.append(("B", next_backward))
            next_backward += 1
        schedule.append(timeline)
    return schedule


def bubble_count(num_stages: int) -> int:
    return 2 * (num_stages - 1)
