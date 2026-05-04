"""Reference solution for L05.8.5 Patch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LATEST_NAME = "latest_checkpointed_iteration.txt"


def atomic_save(payload: dict[str, Any], target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    os.replace(tmp, target)


def _cleanup_partial(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.iterdir():
        if path.suffix == ".tmp" or path.name.endswith(".tmp"):
            try:
                path.unlink()
            except OSError:
                pass


def load_latest(checkpoint_dir: Path) -> dict[str, Any] | None:
    checkpoint_dir = Path(checkpoint_dir)
    _cleanup_partial(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("iter_*.pt"))
    if not candidates:
        return None
    latest_marker = checkpoint_dir / LATEST_NAME
    if latest_marker.exists():
        try:
            marker = json.loads(latest_marker.read_text(encoding="utf-8"))
            target = checkpoint_dir / f"iter_{int(marker['step']):07d}.pt"
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, KeyError, OSError):
            pass
    # fall back to highest-numbered file
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def save_step(
    checkpoint_dir: Path,
    step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    rng_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"iter_{int(step):07d}.pt"
    payload = {
        "step": int(step),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "rng_state": dict(rng_state),
        "extra": dict(extra) if extra else {},
    }
    atomic_save(payload, target)
    atomic_save({"step": int(step)}, checkpoint_dir / LATEST_NAME)
    return target
