#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deployments/runtime_env.sh"
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 {preview|evaluate|run} --confirm-hardware [options]" >&2
  exit 2
fi
mode="$1"
shift
case "$mode" in
  preview|evaluate|run) ;;
  *) echo "Expected preview, evaluate, or run" >&2; exit 2 ;;
esac

# Local-only model loading. Do not inherit unsupported desktop SOCKS settings
# into the offline HF client; these overrides affect only this launch.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

exec "$ROOT/deployments/local_act/run.sh" "$mode" \
  --checkpoint "${OPENARM_SMOLVLA_CHECKPOINT:-$OPENARM_LEROBOT_ROOT/output/train/smolvla_openarm_folding_2gpu/checkpoints/030000/pretrained_model}" \
  --deployment-manifest "${OPENARM_SMOLVLA_MANIFEST:-$ROOT/deployments/local_smolvla/deployment.json}" \
  --num-actions 10 --max-policy-steps 10 "$@"
