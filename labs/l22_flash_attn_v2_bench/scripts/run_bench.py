"""L23 · Run eager vs PyTorch SDPA benchmark grid."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB_DIR / "patch"))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l22_flash_attn_v2_bench"


def _impl():
    requested = os.environ.get("IMPL")
    if requested in {"starter", "reference"}:
        return importlib.import_module(f"{requested}.flash_bench"), requested

    try:
        from starter import flash_bench as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import flash_bench as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _resolve_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    mod, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )

    try:
        import torch  # noqa: F401
    except ImportError:
        write_text(run_dir / "artifacts" / "fallback.txt", "torch missing\n")
        write_text(run_dir / "report.md", f"# {MISSION_ID}\nfallback\n")
        print(run_dir)
        return

    rows = []
    accept_all = True
    for shape in config["shapes"]:
        out = mod.bench_attention(
            seq_len=int(shape["seq_len"]),
            num_heads=int(shape["num_heads"]),
            head_dim=int(shape["head_dim"]),
            causal=bool(config.get("causal", True)),
            device=str(config["device"]),
            dtype=_resolve_dtype(config["dtype"]),
            num_iters=int(config.get("num_iters", 20)),
        )
        out["seq_len"] = int(shape["seq_len"])
        out["num_heads"] = int(shape["num_heads"])
        out["head_dim"] = int(shape["head_dim"])
        rows.append(out)
        append_jsonl(
            run_dir / "metrics.jsonl",
            {"timestamp": utc_now(), "metric_type": "fa_bench", **out},
        )

    csv_lines = ["seq_len,head_dim,eager_ms,flash_ms,speedup,eager_mem_mb,flash_mem_mb,max_abs_diff"]
    for row in rows:
        csv_lines.append(
            f"{row['seq_len']},{row['head_dim']},{row['eager_time_ms']:.3f},{row['flash_time_ms']:.3f},"
            f"{row['speedup']:.3f},{row['peak_mem_eager_mb']:.1f},{row['peak_mem_flash_mb']:.1f},{row['max_abs_diff']:.3e}"
        )
    write_text(run_dir / "artifacts" / "bench.csv", "\n".join(csv_lines) + "\n")

    accept = config.get("acceptance", {})
    for key, value in accept.items():
        if key.startswith("speedup_min_at_seq_"):
            target_seq = int(key.removeprefix("speedup_min_at_seq_"))
            target = float(value)
            for row in rows:
                if row["seq_len"] == target_seq and row["speedup"] < target:
                    accept_all = False
        if key == "max_abs_diff_below":
            for row in rows:
                if row["max_abs_diff"] >= float(value):
                    accept_all = False

    write_json(run_dir / "artifacts" / "bench_summary.json", {"impl": impl_label, "accept": accept_all, "rows": rows})
    write_text(
        run_dir / "report.md",
        f"# {MISSION_ID}\nimpl={impl_label}; rows={len(rows)}; accept={accept_all}\nbench.csv 见 artifacts/。\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
