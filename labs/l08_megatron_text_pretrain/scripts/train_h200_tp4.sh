#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
CONFIG="${2:-configs/h200_tp4.yaml}"
python scripts/run_train.py --config "${CONFIG}" ${RUN_ID:+--run-id "${RUN_ID}"} --mode h200_tp4
