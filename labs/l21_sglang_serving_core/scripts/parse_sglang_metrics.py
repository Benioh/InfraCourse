from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "rows": len(rows),
        "workloads": sorted(
            {row.get("workload") for row in rows if row.get("workload")}
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
