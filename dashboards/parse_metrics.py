from __future__ import annotations

import json
from pathlib import Path


def read_metrics(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("runs/*/*/metrics.jsonl")):
        mission = path.parts[-3]
        run_id = path.parts[-2]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({"mission": mission, "run_id": run_id, **json.loads(line)})
    return rows


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    print(json.dumps(read_metrics(repo_root), indent=2))
