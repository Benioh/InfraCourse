#!/usr/bin/env bash
set -euo pipefail
python scripts/run_slime_lab.py --mode h200_6actor_2rollout ${1:+--run-id "$1"}
