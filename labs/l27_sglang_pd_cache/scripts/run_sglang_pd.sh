#!/usr/bin/env bash
# Print the real SGLang PD launch commands for a profile.
set -euo pipefail
PROFILE="${PROFILE:-h200_qwen}"
LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$LAB_DIR/configs/${PROFILE}.yaml"
python - "$CONFIG" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for key in ("sglang_prefill_command", "sglang_decode_command", "sglang_router_command"):
    if key in config:
        print(f"# {key}")
        print(" ".join(config[key].split()))
        print()
PY
