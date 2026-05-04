#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
python scripts/run_train.py --config configs/4090_debug.yaml ${RUN_ID:+--run-id "${RUN_ID}"} --mode 4090
