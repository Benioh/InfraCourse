"""L03.5 Patch tests · CPU only."""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.megatron_bin")


def _toy_tokenizer(text: str) -> list[int]:
    return [(ord(ch) % 30000) + 1 for ch in text]


def _write_jsonl(path: Path, samples: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps({"text": sample}) + "\n")


def test_writes_bin_idx_pair(tmp_path: Path):
    impl = _impl()
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, ["hello", "world", "megatron"])
    result = impl.text_to_megatron_bin(jsonl, _toy_tokenizer, tmp_path / "out", dtype_str="int32")
    assert (tmp_path / "out.bin").exists()
    assert (tmp_path / "out.idx").exists()
    assert result["n_samples"] == 3
    assert result["dtype"] == "int32"


def test_idx_header_magic_version(tmp_path: Path):
    impl = _impl()
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, ["abc"])
    impl.text_to_megatron_bin(jsonl, _toy_tokenizer, tmp_path / "out", dtype_str="int32")
    with (tmp_path / "out.idx").open("rb") as fh:
        magic = fh.read(8)
        (version,) = struct.unpack("<I", fh.read(4))
        (dtype_code,) = struct.unpack("<B", fh.read(1))
    assert magic == impl.MAGIC
    assert version == impl.VERSION
    assert dtype_code == impl.DTYPE_TO_CODE["int32"]


def test_roundtrip_get_sample(tmp_path: Path):
    impl = _impl()
    samples = ["hello world", "alpha beta gamma", "a"]
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, samples)
    impl.text_to_megatron_bin(jsonl, _toy_tokenizer, tmp_path / "out", dtype_str="int32")
    dataset = impl.IndexedDataset(tmp_path / "out")
    assert len(dataset) == len(samples)
    for i, text in enumerate(samples):
        assert dataset[i] == _toy_tokenizer(text)


def test_total_tokens_matches_concat(tmp_path: Path):
    impl = _impl()
    samples = ["xx", "yyy", "zzzz"]
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, samples)
    result = impl.text_to_megatron_bin(jsonl, _toy_tokenizer, tmp_path / "out", dtype_str="int32")
    expected_bytes = sum(len(s) for s in samples) * 4
    assert (tmp_path / "out.bin").stat().st_size == expected_bytes
    assert result["total_tokens"] == sum(len(s) for s in samples)


def test_empty_lines_skipped(tmp_path: Path):
    impl = _impl()
    jsonl = tmp_path / "in.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "ok"}) + "\n")
        fh.write(json.dumps({"text": ""}) + "\n")
        fh.write(json.dumps({"foo": "no_text_field"}) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"text": "second"}) + "\n")
    result = impl.text_to_megatron_bin(jsonl, _toy_tokenizer, tmp_path / "out", dtype_str="int32")
    assert result["n_samples"] == 2


def test_dtype_uint16_overflow_raises(tmp_path: Path):
    impl = _impl()
    jsonl = tmp_path / "in.jsonl"
    _write_jsonl(jsonl, ["hi"])
    with pytest.raises(ValueError):
        impl.text_to_megatron_bin(
            jsonl, lambda _: [70000], tmp_path / "out", dtype_str="uint16"
        )
