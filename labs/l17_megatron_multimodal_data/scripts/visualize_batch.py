from __future__ import annotations
from pathlib import Path
from _toy_data import MANIFEST, read_jsonl

rows = read_jsonl(MANIFEST)
out = Path("../../data/multimodal_toy/batch_preview.md").resolve()
out.write_text(
    "# Batch Preview\n\n"
    + "\n".join(f"- {r['id']} · {r['task']} · {r['files']}" for r in rows),
    encoding="utf-8",
)
print(out)
