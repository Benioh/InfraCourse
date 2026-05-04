#!/usr/bin/env bash
set -euo pipefail
PROFILE="${PROFILE:-cpu_demo}"
RUN_ID="${1:-}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_DIR"
python scripts/run_preprocess.py --config "configs/${PROFILE}.yaml" ${RUN_ID:+--run-id "${RUN_ID}"}
