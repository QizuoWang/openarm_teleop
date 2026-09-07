#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deployments/runtime_env.sh"
export LD_LIBRARY_PATH="$HIKROBOT_MV3D_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "${1:-}" == "evaluate" ]]; then
  shift
  exec "$OPENARM_DEPLOY_PYTHON" \
    "$ROOT/deployments/local_act/evaluation_server.py" "$@"
fi

exec "$OPENARM_DEPLOY_PYTHON" "$ROOT/deployments/local_act/controller.py" "$@"
