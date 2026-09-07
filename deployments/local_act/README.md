# Local ACT deployment

This deployment uses only `/home/robot/openarm` assets. Its controller uses the same
`openarm_pedestal_vr.yaml`, radian joint convention, right-then-left action
order, and Hikrobot camera serials as dataset collection. The policy boundary
also supports `lerobot-openarm-v1` (left-first degrees) checkpoints, without
changing the controller, raw recorder, hardware limits, or default baseline.

## Newly trained degree-model checkpoint

The adapter reads `config.json` and the run's `openarm_training_manifest.json`.
Keep the checkpoint in its training run, or copy the manifest alongside it.
New-format checkpoints without a matching representation declaration are rejected.
Camera features are renamed to `left_wrist`, `right_wrist`, and `base`; measured
state is converted before the checkpoint preprocessor, and predictions are
converted after its postprocessor. Gripper degree targets become physical
radians (with the left sign restored), not legacy logical commands. Existing
joint/start-delta guards, slew limiting, and the `MOVE` confirmation remain active.

First perform prediction-only preview; this reads hardware but never enables motors:

```bash
cd /home/robot/openarm/dora-openarm-data-collection
./deployments/local_act/run.sh preview --confirm-hardware \
  --checkpoint outputs/act-tshirt-fold-deg-v1-run02/checkpoints/100000/pretrained_model \
  --num-actions 32
```

After reviewing preview output, start a bounded GUI trial:

```bash
./deployments/local_act/run.sh evaluate --confirm-hardware \
  --checkpoint outputs/act-tshirt-fold-deg-v1-run02/checkpoints/100000/pretrained_model \
  --num-actions 32 --max-policy-steps 32
```

Typing `MOVE` homes the arms. Open `http://127.0.0.1:8010`, then use Start & Record
to execute at most 32 steps. This is a bounded motion trial, not a full-fold test.
Only after satisfactory bounded trials, increase `--max-policy-steps` for longer
evaluation. Each launch creates a separate session; session metadata identifies
the checkpoint and model representation. Recorded state and executed actions
remain right-first radians with physical gripper targets. Preview passing does
not establish that the policy will successfully or safely complete a fold.

The adapter change has not been tested or hardware-validated. Unit tests were
added but intentionally not run. No checkpoint weights or baseline defaults
were changed.

## Legacy baseline

The dataset contains upstream logical gripper commands (`right [-1,0]`,
`left [0,1]`), while POS_FORCE feedback and physical travel are approximately
`right [-0.4,0]`, `left [0,0.4]`. Deployment constrains the logical commands to
their trained domains and then saturates them to the configured physical limits.
Arm-joint limit violations remain hard failures.

The checkpoint is the held-out-best step 100000 model, migrated locally to
LeRobot 0.6.2 processor format. ACT predicts and executes its native 85-action
chunk before observing and replanning. The command slew limiter carries the
last command across chunk boundaries instead of resetting to delayed measured
feedback.

Prediction-only hardware preview (opens cameras and reads CAN state, but never
enables motors):

```bash
cd /home/robot/openarm/dora-openarm-data-collection
./deployments/local_act/run.sh preview --confirm-hardware
```

Real motion requires an interactive terminal and a second `MOVE` confirmation:

```bash
./deployments/local_act/run.sh run --confirm-hardware --max-policy-steps 1800
```

Before either real-motion command, both CAN buses must be configured:

```bash
sudo openarm-can-cli -i can0 can_configure
sudo openarm-can-cli -i can1 can_configure
```

For repeated, labeled real-robot evaluations with three live camera views, use
the evaluation UI. Every invocation creates a separate session directory unless
`--session-id` is explicitly supplied:

```bash
./deployments/local_act/run.sh evaluate \
  --confirm-hardware \
  --garment-id tshirt-eval-01 \
  --operator-id operator-01 \
  --max-policy-steps 1800
```

After typing `MOVE` in the terminal, open `http://127.0.0.1:8010`. The UI records
all three cameras, measured joint state, and executed policy actions. Each
rollout must be labeled Success or Failure before the arms can be returned to
the start pose. Raw evaluation sessions are written under
`evaluation_data/sessions/<session-id>/dataset`; incomplete unlabeled rollouts
remain under that dataset's `.pending` directory. After each executed policy
chunk, pending arm actions and observations are atomically checkpointed, so a
graceful interruption preserves Parquet streams and an `episode.yaml` manifest
instead of leaving camera-only data.

Review a completed evaluation session with the synchronized player and relabel
tool:

```bash
./openarm-fold review \
  --raw-root evaluation_data/sessions/<session-id>/dataset \
  --serve \
  --port 8011
```

Convert accepted evaluation rollouts to a separate LeRobot v3 evaluation split:

```bash
./openarm-fold convert \
  --raw-root evaluation_data/sessions/<session-id>/dataset \
  --output-root derived/lerobot/<session-id> \
  --repo-id local/<session-id> \
  --dataset-split evaluation
```

Failed rollout camera/state/action data and labels remain in the raw evaluation
session; conversion intentionally selects accepted episodes only.

Keep your hand beside the physical emergency stop or power disconnect, but do
not press it unless aborting motion. Before inference,
place one T-shirt flat in the same collar orientation, table location, and
initial layout used in the demonstrations; an empty table is outside this
policy's training distribution. Keep the rest of the workspace clear. `Ctrl-C`
immediately disables both arms instead of running the configured park trajectory.
