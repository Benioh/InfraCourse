#!/usr/bin/env bash
set -euo pipefail
python scripts/run_slime_lab.py --mode 4090_smoke ${1:+--run-id "$1"}
