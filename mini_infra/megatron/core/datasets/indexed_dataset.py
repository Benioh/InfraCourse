from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexedSample:
    sample_id: str
    offset: int
    length: int


class IndexedDatasetBuilder:
    """Tiny `.bin/.idx` builder with Megatron-style prefix semantics."""

    def __init__(self, output_prefix: Path) -> None:
        self.output_prefix = output_prefix
        self.bin_path = output_prefix.with_suffix(".bin")
        self.idx_path = output_prefix.with_suffix(".idx")
        self.samples: list[IndexedSample] = []
        self.cursor = 0
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.bin_path.open("wb")

    def add_item(self, sample_id: str, text: str) -> None:
        payload = text.encode("utf-8") + b"\n"
        self.samples.append(IndexedSample(sample_id, self.cursor, len(payload)))
        self._handle.write(payload)
        self.cursor += len(payload)

    def finalize(self) -> None:
        self._handle.close()
        self.idx_path.write_text(
            json.dumps(
                {
                    "format": "mini_megatron_indexed_dataset_v1",
                    "bin": str(self.bin_path),
                    "samples": [sample.__dict__ for sample in self.samples],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class IndexedDataset:
    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.bin_path = prefix.with_suffix(".bin")
        self.idx_path = prefix.with_suffix(".idx")
        payload = json.loads(self.idx_path.read_text(encoding="utf-8"))
        self.samples = [IndexedSample(**sample) for sample in payload["samples"]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        sample = self.samples[index]
        with self.bin_path.open("rb") as handle:
            handle.seek(sample.offset)
            return handle.read(sample.length).decode("utf-8").rstrip("\n")
