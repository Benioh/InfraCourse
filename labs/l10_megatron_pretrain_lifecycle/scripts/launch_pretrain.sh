#!/usr/bin/env bash
set -euo pipefail

# L11 lifecycle launcher.
#   - default profile: cpu_smoke (200 steps, runs anywhere with torch)
#   - PROFILE=4090_debug ./scripts/launch_pretrain.sh
#   - PROFILE=h200_125m  ./scripts/launch_pretrain.sh   # also emits real torchrun cmd
PROFILE="${PROFILE:-cpu_smoke}"
RUN_ID="${1:-}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_DIR"
python scripts/run_lifecycle.py \
    --config "configs/${PROFILE}.yaml" \
    ${RUN_ID:+--run-id "${RUN_ID}"}
