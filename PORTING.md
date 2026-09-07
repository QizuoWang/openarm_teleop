# Deploying on Another PC

## 1. Purpose and boundaries

This guide migrates the standalone OpenArm control, collection, and local
ACT/SmolVLA evaluation software. It does not migrate motor calibration or
authorize autonomous motion. No XDoor installation is required or reused.

The reference policy host is Linux x86-64 with an NVIDIA RTX 3090 (24 GB).
Install a compatible NVIDIA driver for the pinned CUDA 12.8 PyTorch wheels.
Use native Linux SocketCAN, reliable USB/Ethernet camera connections, and the
Hikrobot MV3D RGB-D SDK matching the camera hardware. This backend is not the
generic Hikrobot MVS camera API. Vendor binaries/licenses are not in Git.

Keep these environments separate:

| Environment | Responsibility | Reference Python |
| --- | --- | --- |
| `.venv-collection` | Dora/VR collection, review, recorder | 3.13 |
| `.venv-deploy` | CAN driver, cameras, evaluation GUI | 3.13 |
| Sibling `.venv` | LeRobot conversion, training, ACT inference | 3.12 |
| `.venv-smolvla` | SmolVLA inference dependencies | 3.12 |

Do not copy an existing virtual environment or install LeRobot into the
collection environment. Recreate environments on the destination. The hardware
package pins record the source PC; this transfer has not been executed or
validated on another machine.

## 2. Clone the source and create the workspace

Install Git, `uv`, FFmpeg, CAN utilities, and platform build prerequisites using
your OS/vendor installation instructions. Set up USB/CAN permissions without
running the whole application as root.

```bash
mkdir -p "$HOME/openarm"
cd "$HOME/openarm"
git clone --recurse-submodules https://github.com/QizuoWang/openarm_teleop.git dora-openarm-data-collection
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout fbb811fca92504439792b97d216f0d00c2268382

export OPENARM_WORKSPACE="$HOME/openarm"
cd "$OPENARM_WORKSPACE/dora-openarm-data-collection"
cp deployments/host.env.example .openarm-deploy.env
```

For an existing clone, initialize its pinned dependencies with
`git submodule update --init --recursive`. Local changes must be preserved
before pulling; do not use a hard reset or force push as a migration step.

The expected layout is:

```text
openarm/
├── lerobot/                         # pinned source plus transferred SmolVLA checkpoint
├── .venv/                           # ACT / training environment
├── hikrobot/                        # externally installed MV3D SDK
└── dora-openarm-data-collection/
    ├── .openarm-deploy.env          # local host overrides; ignored by Git
    ├── .venv-deploy/
    ├── .venv-smolvla/
    ├── deployments/act/             # transferred legacy ACT bundle
    ├── deployments/local_smolvla/vlm_assets/
    ├── outputs/                     # transferred new ACT training run
    └── evaluation_data/sessions/    # generated locally
```

## 3. Install the policy and hardware environments

From the collection repository root:

```bash
uv venv --python 3.13 .venv-deploy
uv pip install --python .venv-deploy/bin/python \
  -r deployments/requirements-hardware.lock.txt
uv pip install --python .venv-deploy/bin/python --no-deps \
  -e nodes/dora-openarm -e nodes/dora-hikrobot-rgb-camera

UV_PROJECT_ENVIRONMENT="$OPENARM_WORKSPACE/.venv" \
  uv sync --project "$OPENARM_WORKSPACE/lerobot" --frozen \
  --python 3.12 --extra training --extra dataset --no-dev

UV_PROJECT_ENVIRONMENT="$PWD/.venv-smolvla" \
  uv sync --project "$OPENARM_WORKSPACE/lerobot" --frozen \
  --python 3.12 --extra smolvla --no-dev
```

These commands install software only. They do not test cameras, infer actions,
configure CAN, enable motors, or validate portability. Do not replace the
source revision or normalize using another dataset to resolve a load error.

## 4. Transfer models and external assets

Use an approved file-transfer channel such as SSH/rsync or external storage.
Preserve directory contents, not just `model.safetensors`. Do not upload
recorded people/workspaces or licensed SDK files to Git accidentally.

| Artifact on source PC | Destination relative to the workspace |
| --- | --- |
| Legacy migrated ACT 100K bundle | `dora-openarm-data-collection/deployments/act/act-tshirt-fold-rad-v1-step100000-lerobot062-local/` |
| New degree ACT run, including `openarm_training_manifest.json` | `dora-openarm-data-collection/outputs/act-tshirt-fold-deg-v1-run02/` |
| SmolVLA 30K `pretrained_model/` directory | `lerobot/output/train/smolvla_openarm_folding_2gpu/checkpoints/030000/pretrained_model/` |
| Downloaded SmolVLM configuration/processor/tokenizer assets | `dora-openarm-data-collection/deployments/local_smolvla/vlm_assets/` |
| Hikrobot MV3D SDK and its shared-library dependencies | Vendor installation location; set `HIKROBOT_MV3D_LIB` |

Model bundles need configuration, all model weight files, pre/postprocessor
JSON, normalization tensors, and tokenizer files where present. Retain the ACT
training manifest above its checkpoint directory. Copying the whole training
run is convenient, but optimizer checkpoints are not needed for inference.

The small tracked SmolVLA `deployment.json` identifies step 30000, representation,
task, local runtime, VLM asset revision, and processor/config artifact hashes.
Its paths are relative to the collection repository root. With the layout
above, no path editing is necessary. The full fine-tuned weights include the
VLM; startup does not download substitute base weights. All policy launches
use offline Hugging Face mode.

For another layout, copy `deployment.json` to `deployment.local.json`, change
only its path fields, and set `OPENARM_SMOLVLA_MANIFEST` and
`OPENARM_SMOLVLA_CHECKPOINT` in `.openarm-deploy.env`. Keep the artifact hashes
unchanged for the same checkpoint. A different checkpoint needs its own
reviewed manifest; it is not safe to bypass a hash mismatch.

Raw recordings, derived datasets, and baseline integrity manifests are optional
for inference. Transfer them separately if review, training, or baseline
verification is required. Existing baseline manifests may contain old absolute
paths; retain them as original provenance rather than treating them as portable.

## 5. Configure the host and identify hardware

Edit `.openarm-deploy.env` using [the template](deployments/host.env.example).
Launchers source this trusted local shell file automatically. It must not
contain untrusted code. Defaults derive paths from the repository location.

Confirm the following before any motion:

- `HIKROBOT_MV3D_LIB` points to the installed MV3D library directory, with its
  dependent libraries and vendor device permissions available.
- Right/left/base camera serials identify the physical cameras in the expected
  training views. Defaults are `00DA8573792`, `00DB1265540`, and `00DA8573794`.
- `openarm_pedestal_vr.yaml` identifies the correct CAN interfaces, motor IDs,
  joint signs/offsets, limits, gains, and startup pose. Defaults are right
  `can0`, left `can1`. USB adapter enumeration can change on another PC.
- The same robot retains its valid calibration. A new PC is not a reason to
  reset motor zeros. A different robot must not inherit calibration by assumption.

`OPENARM_ROBOT_CONFIG` overrides the policy evaluator's YAML. Collection still
uses the follower configuration in `dataflow-vr.yaml`; update that descriptor
explicitly if using a different hardware YAML for collection.

Configure the identified CAN buses from an interactive terminal, with the
robot stationary and no competing controller. The current stack uses:

```bash
sudo .venv-deploy/bin/openarm-can-cli -i can0 can_configure
sudo .venv-deploy/bin/openarm-can-cli -i can1 can_configure
```

These are hardware configuration commands, not calibration instructions.
Do not run them against unidentified interfaces. Bus setup alone does not
authorize motion.

## 6. Staged policy startup

Stop the collector and other evaluators first. Place the garment in the
training layout. Keep the physical E-stop accessible. Run these stages
manually; nothing in this migration automatically starts a policy.

SmolVLA prediction-only preview, then a separate bounded GUI trial:

```bash
./deployments/local_smolvla/run.sh preview --confirm-hardware
# Only after reviewing preview output:
./deployments/local_smolvla/run.sh evaluate --confirm-hardware
```

The SmolVLA launcher selects the 30K checkpoint, predicts a 50-action chunk,
executes 10 actions before replanning, and defaults to a 10-step total trial.
Task: `Fold the T-shirt into a compact rectangle.` Inference is synchronous;
the nominal 30 Hz execution rate is not a claim of uninterrupted inference.

Legacy ACT preview and bounded trial:

```bash
./deployments/local_act/run.sh preview --confirm-hardware --num-actions 32
./deployments/local_act/run.sh evaluate --confirm-hardware \
  --num-actions 32 --max-policy-steps 32
```

For the degree ACT checkpoint, append this option to each ACT command:

```bash
--checkpoint outputs/act-tshirt-fold-deg-v1-run02/checkpoints/100000/pretrained_model
```

The GUI is at `http://127.0.0.1:8010`. **Typing `MOVE` homes the arms** before
the GUI starts; Start & Record begins policy execution. The policy-step budget
does not bound homing. Label each rollout Success/Failure before returning to
the start pose. Evaluation sessions are written to `evaluation_data/sessions/`.

After satisfactory bounded trials, explicitly choose a longer budget with
`--max-policy-steps 1800`. Do not increase safety thresholds to work around a
rejection. Use the physical E-stop for emergencies; normal Ctrl-C requests
torque disable and recorder cleanup, which still depend on the process/driver.

## 7. Optional: collection and dataset workflow

Collection uses Dora 0.5.0, separate from the hardware-only policy environment:

```bash
uv venv --python 3.13 .venv-collection
uv pip install --python .venv-collection/bin/python \
  'dora-rs==0.5.0' 'dora-rs-cli==0.5.0' \
  -e nodes/dora-openarm-vr -e nodes/dora-openarm \
  -e nodes/dora-openarm-kinematics -e nodes/dora-openarm-quitter \
  -e nodes/dora-hikrobot-rgb-camera \
  -e nodes/dora-openarm-data-collection-ui -e nodes/dora-openarm-dataset-recorder
source .venv-collection/bin/activate
```

Follow [FOLDING_WORKFLOW.md](FOLDING_WORKFLOW.md) for session initialization and
collection commands. The collector UI uses port 8000; review uses 8001.
`openarm-fold` applies host SDK/camera overrides to its generated session
dataflow. Direct `dora run dataflow-vr.yaml` uses the descriptor's explicit
values instead. Configure the Quest VR application/network separately using
the VR node documentation; cloning the PC repository does not install the headset app.

## 8. Troubleshooting and release status

| Symptom | Action |
| --- | --- |
| `dora: not found` | Activate `.venv-collection`; do not install Dora into the policy environment |
| Port 8000/8010 busy | Identify the existing collector/evaluator and close it normally |
| Missing SDK/shared library | Correct `HIKROBOT_MV3D_LIB`, install vendor dependencies/permissions |
| Missing weights/tokenizer/config | Transfer the complete bundle; startup is deliberately offline |
| Manifest/hash mismatch | Restore the matching artifacts or prepare a separate checkpoint manifest |
| Unit/order/camera mismatch | Resolve training metadata; never bypass the representation adapter |
| Startup delta/joint-limit rejection | Inspect feedback, calibration, pose, and predictions before motion |
| GUI E-stop unavailable | Use the physical E-stop; web controls depend on the PC/network |

Unit tests are supplied but were not run for this publication. No build,
deployment dry-run, GPU inference, camera acquisition, or robot motion was
performed as part of the migration update. Fresh-PC acceptance remains an
operator-controlled sequence: installation, hardware identification, prediction
preview, bounded motion, and finally task evaluation.
