# Pinned LeRobot/ACT environment

Collection and training use separate Python environments. Do not install
LeRobot into the Dora collection environment. Conversion, export verification,
training, and offline policy reporting use this checkout and environment:

The default layout is a sibling `lerobot/` checkout and `.venv/` beside this
repository. The source PC uses `/home/robot/openarm` as that parent directory.
Use `.openarm-deploy.env` to override paths; see [PORTING.md](PORTING.md).

## Create the environment

The supported checkout is LeRobot 0.6.2 at commit
`fbb811fca92504439792b97d216f0d00c2268382`. To reproduce it from a fresh clone,
use Python 3.12 and the checked-in dependency lock:

```bash
export OPENARM_WORKSPACE="$HOME/openarm"
cd "$OPENARM_WORKSPACE/lerobot"
git checkout fbb811fca92504439792b97d216f0d00c2268382
UV_PROJECT_ENVIRONMENT="$OPENARM_WORKSPACE/.venv" \
  uv sync --frozen --python 3.12 --extra training --extra dataset --no-dev
```

`requirements-lerobot.lock.txt` records the same source revision for provenance.
The workflow additionally checks the imported module path, package version,
Git revision, clean tracked worktree, and the checkout's `uv.lock`.

## GPU readiness boundary

This workflow does not repair NVIDIA device access. Before training, explicitly
run:

```bash
./openarm-fold preflight --training
```

Do not start ACT training until every training preflight item reports `OK`.
This check does not enable the robot or authorize deployment.

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
