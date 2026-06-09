"""Reference solution for L08 Patch."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np

MAGIC = b"MGTRNIDX"
VERSION = 1
DTYPE_TO_CODE = {"int32": 1, "uint16": 2, "int64": 3}
CODE_TO_DTYPE = {value: key for key, value in DTYPE_TO_CODE.items()}
DTYPE_TO_NUMPY = {"int32": np.int32, "uint16": np.uint16, "int64": np.int64}
DTYPE_TO_BYTES = {"int32": 4, "uint16": 2, "int64": 8}
DTYPE_MAX = {"int32": 2**31 - 1, "uint16": 2**16 - 1, "int64": 2**63 - 1}


def text_to_megatron_bin(
    jsonl_path: Path,
    tokenizer: Callable[[str], list[int]],
    output_prefix: Path,
    dtype_str: str = "int32",
) -> dict:
    if dtype_str not in DTYPE_TO_CODE:
        raise ValueError(f"unsupported dtype: {dtype_str}")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    nbytes = DTYPE_TO_BYTES[dtype_str]
    np_dtype = DTYPE_TO_NUMPY[dtype_str]
    max_token = DTYPE_MAX[dtype_str]

    offsets: list[int] = []
    lengths: list[int] = []
    cursor = 0
    with bin_path.open("wb") as bin_file, Path(jsonl_path).open("r", encoding="utf-8") as jsonl:
        for raw in jsonl:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload.get("text", "")
            if not text:
                continue
            tokens = list(tokenizer(text))
            if not tokens:
                continue
            for token in tokens:
                if int(token) < 0 or int(token) > max_token:
                    raise ValueError(
                        f"token id {token} out of range for dtype {dtype_str}"
                    )
            arr = np.asarray(tokens, dtype=np_dtype)
            bin_file.write(arr.tobytes())
            offsets.append(cursor)
            lengths.append(len(tokens))
            cursor += len(tokens) * nbytes

    n_samples = len(lengths)
    total_tokens = sum(lengths)
    with idx_path.open("wb") as idx_file:
        idx_file.write(MAGIC)
        idx_file.write(struct.pack("<I", VERSION))
        idx_file.write(struct.pack("<B", DTYPE_TO_CODE[dtype_str]))
        idx_file.write(struct.pack("<Q", n_samples))
        idx_file.write(struct.pack("<Q", total_tokens))
        idx_file.write(np.asarray(offsets, dtype=np.uint64).tobytes())
        idx_file.write(np.asarray(lengths, dtype=np.uint64).tobytes())

    return {
        "n_samples": n_samples,
        "total_tokens": total_tokens,
        "bin_path": str(bin_path),
        "idx_path": str(idx_path),
        "dtype": dtype_str,
    }


class IndexedDataset:
    def __init__(self, prefix: Path) -> None:
        prefix = Path(prefix)
        idx_path = prefix.with_suffix(".idx")
        bin_path = prefix.with_suffix(".bin")
        with idx_path.open("rb") as idx_file:
            magic = idx_file.read(8)
            if magic != MAGIC:
                raise ValueError(f"bad magic: {magic!r}")
            (version,) = struct.unpack("<I", idx_file.read(4))
            if version != VERSION:
                raise ValueError(f"unsupported version: {version}")
            (dtype_code,) = struct.unpack("<B", idx_file.read(1))
            (self._n,) = struct.unpack("<Q", idx_file.read(8))
            (self._total,) = struct.unpack("<Q", idx_file.read(8))
            self._dtype_str = CODE_TO_DTYPE[dtype_code]
            offsets = np.frombuffer(idx_file.read(self._n * 8), dtype=np.uint64).copy()
            lengths = np.frombuffer(idx_file.read(self._n * 8), dtype=np.uint64).copy()
        self._offsets = offsets
        self._lengths = lengths
        self._np_dtype = DTYPE_TO_NUMPY[self._dtype_str]
        self._nbytes = DTYPE_TO_BYTES[self._dtype_str]
        self._mmap = np.memmap(bin_path, dtype=self._np_dtype, mode="r")

    @property
    def dtype(self) -> str:
        return self._dtype_str

    @property
    def total_tokens(self) -> int:
        return int(self._total)

    def __len__(self) -> int:
        return int(self._n)

    def __getitem__(self, index: int) -> list[int]:
        if index < 0 or index >= self._n:
            raise IndexError(index)
        start = int(self._offsets[index]) // self._nbytes
        length = int(self._lengths[index])
        return [int(value) for value in self._mmap[start : start + length]]
