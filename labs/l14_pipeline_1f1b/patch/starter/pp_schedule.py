"""L05.7 Patch · 1F1B Pipeline Schedule generator."""

from __future__ import annotations


def make_1f1b_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple[str, int]]]:
    """Return per-stage 1F1B timelines, each entry an (op, microbatch_idx)."""
    # TODO(student): validate inputs and raise ValueError when num_microbatches < num_stages
    # TODO(student): for each stage s in [0, num_stages):
    #   warmup = num_stages - s - 1
    #   emit warmup forward ops on microbatches [0..warmup-1]
    #   then steady = num_microbatches - warmup iterations of (forward, backward)
    #   then cooldown = warmup backward ops
    raise NotImplementedError("L05.7: implement make_1f1b_schedule")


def bubble_count(num_stages: int) -> int:
    """Pipeline bubble length under the standard (non-interleaved) 1F1B schedule."""
    # TODO(student): return 2 * (num_stages - 1)
    raise NotImplementedError("L05.7: implement bubble_count")
