from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_infra.observability.evidence import scan
from mini_infra.observability.io import ROOT, write_json, write_text


REQUIRED_HALF_MISSIONS = [
    "l01_7_gpu_kernel",
    "l04_5_long_context_cp",
    "l05_5_moe_ep",
    "l06_3_data_engineering",
    "l08_5_quant_serving",
    "l08_7_spec_decode",
]


def build_delivery(source: Path, output: Path) -> dict[str, int]:
    rows = scan(source)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evidence_index.json", rows)
    lines = [
        "# MiniInfra Delivery",
        "",
        "| Run | Metrics | Report | Artifacts |",
        "|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_dir']}` | {row['metric_count']} | {row['has_report']} | {row['artifact_count']} |"
        )
    lines.extend(["", "## Required Half Missions", ""])
    present = {
        row["run_dir"].split("/")[1]
        if row["run_dir"].startswith("runs/") and len(row["run_dir"].split("/")) > 2
        else row["run_dir"]
        for row in rows
    }
    for mission in REQUIRED_HALF_MISSIONS:
        status = "present" if mission in present else "missing"
        lines.append(f"- `{mission}`: {status}")
    write_text(output / "final_readme.md", "\n".join(lines) + "\n")
    write_text(
        output / "risk_register.md",
        "# MiniInfra Risk Register\n\n"
        "- 缺 command/config/report 的 run 不能作为交付证据。\n"
        "- simulated backend 不能作为真实性能结论。\n"
        "- L12 evidence index 必须检查 L01.7/L04.5/L05.5/L06.3/L08.5/L08.7 六个新增半关是否 present。\n",
    )
    return {"run_count": len(rows), "artifact_count": len(list(output.rglob("*")))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniInfra delivery bundle")
    parser.add_argument("--source", default=str(ROOT / "runs"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "mini_infra" / "delivery"))
    args = parser.parse_args()
    print(
        json.dumps(
            build_delivery(Path(args.source), Path(args.output)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
