# Local SmolVLA deployment

Uses the copied 30K checkpoint at
`/home/robot/openarm/lerobot/output/train/smolvla_openarm_folding_2gpu/checkpoints/030000/pretrained_model`.
No weights, ACT defaults, joint limits, or pedestal startup poses are changed.
No XDoor code/assets are used. This implementation has not been tested or
hardware-validated; installation and asset download are not an inference test.

## Runtime

- Hardware/GUI: existing `.venv-deploy` and guarded controller.
- Policy: isolated `.venv-smolvla`, installed from the local LeRobot checkout's
  frozen lock with the `smolvla` extra. ACT's policy environment is unchanged.
- Model: full fine-tuned local weights, checkpoint tokenizer and normalization.
  VLM architecture/processor assets are downloaded locally at the revision in
  `deployment.json`; no base weights are downloaded at launch. SmolVLA weight
  loading is strict so incompatible/incomplete checkpoints fail before motion.
- Model state/actions: left-first degrees and calibrated gripper degrees.
  Controller: right-first radians and physical gripper targets, as for degree ACT.
- Task: `Fold the T-shirt into a compact rectangle.`
- The model predicts 50 actions using 10 denoising steps; the executor uses 10
  actions per inference by default, matching saved `n_action_steps=10`, at 30 Hz.
  Inference is synchronous; this does not promise lag-free operation.

## Commands

Stop the collector or other robot evaluator before opening the cameras/CAN here.
Ensure the arms/table are clear and the physical E-stop is immediately accessible.
Start with the same shirt layout and fixed camera positions as the training data.

Prediction-only preview (motors never enabled):

```bash
cd /home/robot/openarm/dora-openarm-data-collection
./deployments/local_smolvla/run.sh preview --confirm-hardware
```

After reviewing preview output, a bounded 10-step GUI trial:

```bash
./deployments/local_smolvla/run.sh evaluate --confirm-hardware
```

Typing `MOVE` homes the arms. Open `http://127.0.0.1:8010`; Start & Record runs
at most 10 policy steps. Success/Failure labels and camera/state/action recording
work as in ACT. Sessions go to `evaluation_data/sessions/smolvla-eval-*/dataset`.
The GUI displays the model name, checkpoint and instruction.

Only after satisfactory bounded trials, increase the rollout budget:

```bash
./deployments/local_smolvla/run.sh evaluate --confirm-hardware --max-policy-steps 1800
```

This is a maximum of 1800 executed steps, not guaranteed completion. Startup
homing is separate from the policy-step budget. The first-target/joint-limit
guards and command slew limiting are retained. Do not increase safety limits
to force an out-of-range prediction through.

The manifest is pinned to this exact checkpoint and processor artifacts. To
deploy another training checkpoint, create a separately reviewed manifest;
do not silently repoint this one or replace its processor statistics.
