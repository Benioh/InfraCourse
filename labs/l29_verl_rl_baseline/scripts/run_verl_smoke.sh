#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-}"
python scripts/run_verl_lab.py ${RUN_ID:+--run-id "${RUN_ID}"}
