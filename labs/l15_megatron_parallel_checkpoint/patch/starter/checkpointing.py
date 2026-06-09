"""L16 Patch · Megatron-shaped distributed checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any


FORMAT = "mini_megatron_distributed_checkpoint_v1"


class CheckpointError(ValueError):
    """Raised when checkpoint metadata is missing or incompatible."""


def save_checkpoint(
    output_dir: Path,
    iteration: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    parallel_state: dict[str, Any],
) -> dict[str, str]:
    """Write a Megatron-shaped JSON checkpoint and latest marker."""
    # TODO(student): create output_dir.
    # TODO(student): write iter_XXXXXXX.json with format/iteration/states.
    # TODO(student): write latest_checkpointed_iteration.txt.
    raise NotImplementedError("L16: implement save_checkpoint")


def load_checkpoint(
    checkpoint_dir: Path,
    expected_parallel_state: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load the latest checkpoint and validate parallel-state compatibility."""
    # TODO(student): read latest marker and checkpoint JSON.
    # TODO(student): validate FORMAT.
    # TODO(student): compare expected_parallel_state against checkpoint parallel_state.
    # TODO(student): strict=True raises CheckpointError; strict=False returns warnings.
    raise NotImplementedError("L16: implement load_checkpoint")
