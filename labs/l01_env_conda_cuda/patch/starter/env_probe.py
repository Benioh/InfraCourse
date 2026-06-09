"""L01 patch starter — Environment Probe.

Implement three functions so a downstream lab can ask "is this machine ready"
and get a JSON answer instead of a guess.

Read patch/task.md before you start. Tests live in
labs/l01_env_conda_cuda/patch/tests/test_patch.py and run on CPU only.
"""
from __future__ import annotations

import os
import sys
from typing import Any


def collect_python_env() -> dict[str, Any]:
    """Return a snapshot of the current Python / launcher environment.

    Required keys (see task.md for exact semantics):

    - python_version: str           e.g. "3.11.9"
    - python_executable: str        sys.executable
    - pythonpath: list[str]         os.pathsep-split, drop empties
    - cuda_visible_devices: str|None  None when env var is unset, "" when set-but-empty
    - local_rank: int|None          int(LOCAL_RANK) or None
    - world_size: int|None          int(WORLD_SIZE) or None

    Do NOT import torch — this function must work on a clean Python install.
    """
    # TODO: build the snapshot dict described above and return it.
    raise NotImplementedError


def parse_local_rank_mapping(env: dict[str, Any], local_rank: int) -> dict[str, Any]:
    """Resolve which physical GPU `local_rank` will land on.

    Inputs:
        env: a dict from collect_python_env() (must contain cuda_visible_devices).
        local_rank: a non-negative int.

    Returns dict with keys:
        - local_rank: pass through
        - visible_devices: list[int]
            - [] when CUDA_VISIBLE_DEVICES is None or empty string (= "see all")
            - else parsed comma-separated ints
        - physical_device: int
            - == local_rank when visible_devices is []
            - else visible_devices[local_rank]

    Raises:
        ValueError: if local_rank < 0.
        ValueError: if CUDA_VISIBLE_DEVICES contains a non-int token
                    (message must mention "CUDA_VISIBLE_DEVICES").
        IndexError: if local_rank is past the end of a non-empty visible_devices
                    (message must contain "local_rank=", "visible_devices=", "out of range").
    """
    # TODO: implement the validation + mapping logic above.
    raise NotImplementedError


def summarize_drift(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Diff two collect_python_env() snapshots, return human-readable lines.

    - Only mention keys whose values changed (and keys that appear on only
      one side — describe with "<missing>" placeholder).
    - Each line is exactly:  "<key>: <baseline_val> → <current_val>"
    - Sorted alphabetically by key.
    - Identical inputs → return [].
    """
    # TODO: implement the alphabetical, "key: a → b" diff.
    raise NotImplementedError
