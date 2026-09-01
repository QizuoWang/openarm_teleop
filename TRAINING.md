# Pinned LeRobot/ACT environment

Collection and training use separate Python environments. Do not install
LeRobot into the Dora collection environment.

## Create the environment

Use Python 3.11:

```bash
python3.11 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot.lock.txt
```

`requirements-lerobot.lock.txt` pins the exact official LeRobot source revision
referenced by the Dataset v3 documentation. Its transitive CUDA/PyTorch wheels
cannot be frozen until GPU device access is restored; after the first approved
installation, capture that resolved wheel set before treating the environment
as fully reproducible. The environment is local and is not created
automatically.

## Current GPU readiness boundary

The host PCI and driver metadata identify an RTX 3090, but the current shell has
no `/dev/nvidia*` devices and `nvidia-smi` cannot communicate with the GPU.
This workflow does not repair NVIDIA device access. Diagnose that separately,
then explicitly run:

```bash
./openarm-fold preflight --training
```

Do not start ACT training until every training preflight item reports `OK`.

## Outputs and provenance

`./openarm-fold train` records:

- resolved dataset root and repository identifier;
- Git revision;
- pinned ACT profile and resolved chunk size;
- exact launch command;
- a deployment authorization flag fixed to `false`.

LeRobot writes its normal metrics, logs, processors, and checkpoints below the
selected output directory. Upload to a private Hugging Face repository is
optional and must be performed as a separate explicit action.
