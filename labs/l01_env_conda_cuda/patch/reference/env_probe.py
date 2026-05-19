"""Reference solution for L00 patch — env probe."""
from __future__ import annotations

import os
import sys
from typing import Any


def collect_python_env() -> dict[str, Any]:
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = [seg for seg in raw_pythonpath.split(os.pathsep) if seg]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")  # None when unset, "" when set-empty

    def _maybe_int(name: str) -> int | None:
        raw = os.environ.get(name)
        return int(raw) if raw is not None and raw != "" else None

    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "pythonpath": pythonpath,
        "cuda_visible_devices": cvd,
        "local_rank": _maybe_int("LOCAL_RANK"),
        "world_size": _maybe_int("WORLD_SIZE"),
    }


def parse_local_rank_mapping(env: dict[str, Any], local_rank: int) -> dict[str, Any]:
    if local_rank < 0:
        raise ValueError(f"local_rank must be >= 0, got {local_rank}")

    cvd = env.get("cuda_visible_devices")
    if cvd is None or cvd == "":
        visible: list[int] = []
    else:
        try:
            visible = [int(seg.strip()) for seg in cvd.split(",") if seg.strip() != ""]
        except ValueError as exc:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES contains a non-integer token: {cvd!r}"
            ) from exc

    if visible:
        if local_rank >= len(visible):
            raise IndexError(
                f"local_rank={local_rank} visible_devices={visible} out of range"
            )
        physical = visible[local_rank]
    else:
        physical = local_rank

    return {
        "local_rank": local_rank,
        "visible_devices": visible,
        "physical_device": physical,
    }


_MISSING = "<missing>"


def summarize_drift(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    keys = sorted(set(baseline) | set(current))
    diffs: list[str] = []
    for key in keys:
        a = baseline.get(key, _MISSING)
        b = current.get(key, _MISSING)
        if a == b:
            continue

        def fmt(value: Any) -> str:
            if value is _MISSING:
                return _MISSING
            if isinstance(value, list):
                return repr(value)
            return str(value)

        diffs.append(f"{key}: {fmt(a)} → {fmt(b)}")
    return diffs
