"""Reference solution for L16 Patch."""

from __future__ import annotations

import json
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
    if iteration < 0:
        raise CheckpointError("iteration must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "iteration": int(iteration),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": dict(scheduler_state),
        "parallel_state": dict(parallel_state),
    }
    checkpoint_path = output_dir / f"iter_{iteration:07d}.json"
    checkpoint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    latest_marker = output_dir / "latest_checkpointed_iteration.txt"
    latest_marker.write_text(str(iteration), encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "latest_marker": str(latest_marker),
    }


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        raise CheckpointError("missing latest_checkpointed_iteration.txt")
    try:
        iteration = int(marker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise CheckpointError("latest checkpoint marker is not an integer") from exc
    path = checkpoint_dir / f"iter_{iteration:07d}.json"
    if not path.exists():
        raise CheckpointError(f"checkpoint file not found: {path.name}")
    return path


def load_checkpoint(
    checkpoint_dir: Path,
    expected_parallel_state: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    path = _latest_checkpoint_path(checkpoint_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise CheckpointError("unsupported checkpoint format")

    warnings: list[str] = []
    actual = payload.get("parallel_state") or {}
    for key, expected in (expected_parallel_state or {}).items():
        actual_value = actual.get(key)
        if actual_value != expected:
            message = f"parallel_state mismatch for {key}: expected {expected}, got {actual_value}"
            if strict:
                raise CheckpointError(message)
            warnings.append(message)

    payload["checkpoint_path"] = str(path)
    payload["warnings"] = warnings
    return payload
