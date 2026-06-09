"""L17 Patch · Crash-safe checkpoint atomic save + resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LATEST_NAME = "latest_checkpointed_iteration.txt"


def atomic_save(payload: dict[str, Any], target: Path) -> None:
    """Atomically write JSON-serializable payload to target."""
    # TODO(student): write payload to <target>.tmp using json.dumps
    # TODO(student): flush + os.fsync
    # TODO(student): os.replace(<target>.tmp, target) — POSIX-atomic same-FS rename
    raise NotImplementedError("L17: implement atomic_save")


def load_latest(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Return latest committed checkpoint payload, or None.

    Must clean up any dangling .tmp files left behind by previous crashes.
    """
    # TODO(student): unlink any iter_*.pt.tmp files left in checkpoint_dir
    # TODO(student): if LATEST_NAME exists and points to a real iter_*.pt, load that
    # TODO(student): otherwise return None
    raise NotImplementedError("L17: implement load_latest")


def save_step(
    checkpoint_dir: Path,
    step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    rng_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Idempotent crash-safe save."""
    # TODO(student): make sure checkpoint_dir exists
    # TODO(student): build payload {"step", "model_state", "optimizer_state", "rng_state", "extra"}
    # TODO(student): target = checkpoint_dir / f"iter_{step:07d}.pt"
    # TODO(student): atomic_save(payload, target)
    # TODO(student): atomic_save({"step": step}, checkpoint_dir / LATEST_NAME)
    # TODO(student): return target
    raise NotImplementedError("L17: implement save_step")
