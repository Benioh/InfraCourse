#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
python scripts/run_train.py --config configs/h200_tp1.yaml ${RUN_ID:+--run-id "${RUN_ID}"} --mode h200_tp1
