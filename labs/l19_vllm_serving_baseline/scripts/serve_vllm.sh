#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-}"
python scripts/run_vllm_lab.py --mode serve ${RUN_ID:+--run-id "${RUN_ID}"}
