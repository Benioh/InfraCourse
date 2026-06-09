"""L08 Patch · Convert JSONL text to a Megatron-shaped IndexedDataset."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

MAGIC = b"MGTRNIDX"
VERSION = 1
DTYPE_TO_CODE = {"int32": 1, "uint16": 2, "int64": 3}
DTYPE_TO_BYTES = {"int32": 4, "uint16": 2, "int64": 8}
DTYPE_MAX = {"int32": 2**31 - 1, "uint16": 2**16 - 1, "int64": 2**63 - 1}


def text_to_megatron_bin(
    jsonl_path: Path,
    tokenizer: Callable[[str], list[int]],
    output_prefix: Path,
    dtype_str: str = "int32",
) -> dict:
    """Stream a JSONL file to <prefix>.bin / <prefix>.idx and return a summary."""
    # TODO(student): validate dtype_str against DTYPE_TO_CODE
    # TODO(student): open <prefix>.bin for writing in binary mode
    # TODO(student): walk the JSONL, skip blank/missing "text", tokenize, range-check tokens
    # TODO(student): write tokens to .bin in the requested dtype (use struct.pack or array)
    # TODO(student): record offsets[] and lengths[] as you go
    # TODO(student): write <prefix>.idx with MAGIC | VERSION | dtype_code | n_samples | total_tokens
    #               followed by offsets[] (uint64) and lengths[] (uint64), all little-endian
    # TODO(student): return {"n_samples": ..., "total_tokens": ..., "bin_path": ..., "idx_path": ...}
    raise NotImplementedError("L08: implement text_to_megatron_bin")


class IndexedDataset:
    """Memory-mapped reader for the Megatron-shaped IndexedDataset format."""

    def __init__(self, prefix: Path) -> None:
        # TODO(student): open <prefix>.idx, validate MAGIC + VERSION
        # TODO(student): read dtype_code -> dtype_str; n_samples, total_tokens
        # TODO(student): read offsets[] and lengths[]
        # TODO(student): np.memmap the .bin file as a 1D array of dtype_str
        raise NotImplementedError("L08: implement IndexedDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> list[int]:
        # TODO(student): use offsets[index] / dtype bytes -> start element index
        # TODO(student): slice mmap view of length lengths[index]
        # TODO(student): return as list[int]
        raise NotImplementedError
