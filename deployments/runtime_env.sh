#!/usr/bin/env bash
# Source from a launcher; never enables hardware or installs dependencies.
OPENARM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$OPENARM_REPO_ROOT/.openarm-deploy.env" ]]; then
  set -a
  source "$OPENARM_REPO_ROOT/.openarm-deploy.env"
  set +a
fi
export OPENARM_WORKSPACE="${OPENARM_WORKSPACE:-$(dirname "$OPENARM_REPO_ROOT")}"
export OPENARM_LEROBOT_ROOT="${OPENARM_LEROBOT_ROOT:-$OPENARM_WORKSPACE/lerobot}"
export OPENARM_POLICY_PYTHON="${OPENARM_POLICY_PYTHON:-$OPENARM_WORKSPACE/.venv/bin/python}"
export OPENARM_DEPLOY_PYTHON="${OPENARM_DEPLOY_PYTHON:-$OPENARM_REPO_ROOT/.venv-deploy/bin/python}"
export OPENARM_ROBOT_CONFIG="${OPENARM_ROBOT_CONFIG:-$OPENARM_REPO_ROOT/openarm_pedestal_vr.yaml}"
export HIKROBOT_MV3D_LIB="${HIKROBOT_MV3D_LIB:-$OPENARM_WORKSPACE/hikrobot/Mv3dRgbdSDK_ROS2/src/hik_rgbd/lib}"
