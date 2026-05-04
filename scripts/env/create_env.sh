#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-}"

if [[ -z "${ENV_NAME}" ]]; then
  echo "usage: bash scripts/env/create_env.sh <env-name>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/envs/${ENV_NAME}.yaml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "environment file not found: ${ENV_FILE}"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH"
  echo "Create environment manually from ${ENV_FILE}"
  exit 0
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env already exists: ${ENV_NAME}"
  exit 0
fi

echo "creating conda environment ${ENV_NAME} from ${ENV_FILE}"
conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
