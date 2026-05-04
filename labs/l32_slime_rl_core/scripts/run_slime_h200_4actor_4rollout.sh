#!/usr/bin/env bash
set -euo pipefail
python scripts/run_slime_lab.py --mode h200_4actor_4rollout ${1:+--run-id "$1"}
