from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_infra.observability.io import ROOT, read_jsonl, write_json


def summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "run_dir": (
            str(run_dir.relative_to(ROOT))
            if run_dir.is_relative_to(ROOT)
            else str(run_dir)
        ),
        "has_command": (run_dir / "command.sh").exists(),
        "has_config": (run_dir / "config.resolved.yaml").exists(),
        "has_report": (run_dir / "report.md").exists(),
        "metric_count": len(metrics),
        "latest_metric": metrics[-1] if metrics else None,
        "artifact_count": (
            len(list((run_dir / "artifacts").rglob("*")))
            if (run_dir / "artifacts").exists()
            else 0
        ),
    }


def scan(root: Path) -> list[dict[str, Any]]:
    return [summarize_run(path) for path in sorted(root.glob("*/*")) if path.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan MiniInfra evidence runs")
    parser.add_argument("--root", default=str(ROOT / "runs" / "mini_infra"))
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = scan(Path(args.root))
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
