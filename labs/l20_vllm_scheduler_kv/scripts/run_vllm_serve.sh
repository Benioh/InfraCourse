#!/usr/bin/env bash
# Print and (best-effort) execute the real `vllm serve` command for a given profile.
set -euo pipefail
PROFILE="${PROFILE:-4090_qwen}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$LAB_DIR/configs/${PROFILE}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[run_vllm_serve] missing config: $CONFIG" >&2
    exit 2
fi
CMD=$(python -c "import sys, yaml; print(' '.join(yaml.safe_load(open(sys.argv[1]))['vllm_serve_command'].split()))" "$CONFIG")
echo "==> Will execute:"
echo "    $CMD"
if python -c "import vllm" 2>/dev/null; then
    eval "$CMD"
else
    echo "[run_vllm_serve] vllm not installed; printed command above for manual launch." >&2
fi
