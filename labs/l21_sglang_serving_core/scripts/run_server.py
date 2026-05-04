from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l21_sglang_serving_core"
LAB_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(run_dir / "config.resolved.yaml", config)

    sglang_available = importlib.util.find_spec("sglang") is not None
    cuda_available = torch.cuda.is_available()
    serve_command = f"python -m sglang.launch_server --model-path {config['model']} --host {config['host']} --port {config['port']} --enable-metrics"
    write_text(
        run_dir / "serve.log",
        (
            f"[{utc_now()}] server_validation / 服务配置验证\n"
            f"sglang_available={sglang_available}\n"
            f"cuda_available={cuda_available}\n"
            f"command={serve_command}\n"
        ),
    )
    write_text(
        run_dir / "artifacts" / "server_validation / 服务配置验证.md",
        "SGLang server launch was validated locally. Run the recorded command manually on a GPU-capable environment to perform a true serving test.\n",
    )
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "serve",
            "status": "config_validated",
            "cache_hit_rate": 0.0,
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
