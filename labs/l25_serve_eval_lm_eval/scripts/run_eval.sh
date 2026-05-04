#!/usr/bin/env bash
set -euo pipefail
PROFILE="${PROFILE:-local_stub}"
RUN_ID="${1:-}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_DIR"
python scripts/run_eval.py --config "configs/${PROFILE}.yaml" ${RUN_ID:+--run-id "${RUN_ID}"}
