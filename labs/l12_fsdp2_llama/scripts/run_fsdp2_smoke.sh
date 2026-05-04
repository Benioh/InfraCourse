#!/usr/bin/env bash
set -euo pipefail
PROFILE="${PROFILE:-cpu_dryrun}"
RUN_ID="${1:-}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_DIR"
if [ "$PROFILE" = "cpu_dryrun" ]; then
    python scripts/run_fsdp2_smoke.py --config "configs/${PROFILE}.yaml" ${RUN_ID:+--run-id "${RUN_ID}"}
else
    NPROC="${NPROC:-8}"
    torchrun --nproc_per_node="$NPROC" --standalone \
        scripts/run_fsdp2_smoke.py --config "configs/${PROFILE}.yaml" ${RUN_ID:+--run-id "${RUN_ID}"}
fi
