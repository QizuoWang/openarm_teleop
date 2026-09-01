# OpenArm T-shirt folding workflow

This workflow keeps lossless robot data separate from derived LeRobot data.
It does not authorize real-robot policy deployment.

## Safety boundary

- Starting Dora leaves both arms stopped.
- Passing data preflight is not permission to move the arms.
- Hold **X** only after the workspace is clear and the monitor says data
  preflight passed. The arms must still complete their normal gripper-gated
  alignment before collection becomes ready.
- A short **X** press pauses arm commands immediately.
- Park is locked. It remains unavailable until a park pose and trajectory have
  been separately approved on the real cell.
- Policy deployment and policy-driven motion are outside this workflow.

## Pilot protocol

- Task: `Fold the T-shirt into a compact rectangle.`
- Start front-down with the collar nearest the robot.
- Fold left sleeve/side, right sleeve/side, bottom third toward the collar, then
  toward the collar once more.
- Keep initial translation within approximately 5 cm, yaw within 10 degrees,
  and wrinkles minor.
- Accept only a stable compact rectangle with sleeves contained and no more
  than approximately 3 cm protrusion.
- Collect four 25-episode training-shirt batches. Reserve one or two other
  shirts for physical evaluation and never collect training demonstrations
  with them.

## 1. Prepare a collection session

```bash
./openarm-fold collect --prepare-only \
  --session-id fold-pilot-shirt01 \
  --operator-id operator-01 \
  --garment-id tshirt-train-01 \
  --dataset-split train \
  --size medium \
  --material cotton \
  --color blue \
  --target 25
```

This writes `.openarm-fold/session.yaml`. Garment metadata is selected once per
batch, not once per episode.

## 2. Inspect static readiness

```bash
./openarm-fold preflight
```

This is read-only. It checks required files, the session profile, Dora, and disk
headroom. Live camera, Quest, recorder, and arm status appear on the PC monitor
after Dora starts.

## 3. Collect

```bash
./openarm-fold collect --resume
```

Open `http://127.0.0.1:8000` on the PC monitor.

Quest controls:

| State | Control | Result |
|---|---|---|
| Ready | A | Start episode when every hard gate passes |
| Recording | A | Accept and finalize |
| Recording | B | Pause arms and open retained-failure menu |
| Failure menu | Left joystick, A | Select and confirm failure reason |
| Any | X | Pause arms immediately |
| Idle after preflight | Hold X 1.5 s | Enable arms; alignment is still required |
| Recording | Hold Y 1.5 s | Quarantine operator-error episode |
| Idle | Hold Y 1.5 s | Request park; currently locked |
| Idle | Hold B 1.5 s | Quit collection |

Accepted and failed episodes are published atomically. Cancelled or interrupted
episodes move to `folding_data/dataset/quarantine/`. A restart chooses the next
unused episode and never appends to a partial motion trajectory.

## 4. Review every 25 accepted episodes

```bash
./openarm-fold review
./openarm-fold report
```

Outputs:

- `reports/folding/episodes.html`: first/middle/final ceiling frames.
- `reports/folding/dataset-report.json`: complete-episode split, garment
  counts, failure reasons, duration distribution, batch gate, and the ACT chunk
  candidate after 25 accepted episodes.

Pause collection when rejection exceeds 10 percent, a required stream fails,
or the canonical fold drifts. Duration is only a warning; the first batch's
95th percentile becomes the soft threshold.

## 5. Convert complete episodes to LeRobotDataset v3

Set up the pinned environment in [TRAINING.md](TRAINING.md), then export the
train and validation splits separately:

```bash
./openarm-fold convert \
  --dataset-split train \
  --repo-id local/openarm-tshirt-fold \
  --output-root derived/lerobot/openarm-tshirt-fold-train

./openarm-fold convert \
  --dataset-split validation \
  --repo-id local/openarm-tshirt-fold-validation \
  --output-root derived/lerobot/openarm-tshirt-fold-validation
```

Conversion uses a fixed 30 Hz overlapping timeline:

- nearest original camera frame within 25 ms;
- linearly interpolated measured joint state;
- most recent command at or before the frame timestamp;
- no fabricated camera, state, or action samples.

Raw 640 x 360 JPEGs are retained. Derived camera streams are H.264 MP4. The
converter refuses to overwrite an existing output and writes an alignment
manifest.

## 6. Train and report

Copy the chunk candidate from the 25-episode report into the command or the
pinned profile:

```bash
./openarm-fold preflight --training

./openarm-fold train \
  --dataset-root derived/lerobot/openarm-tshirt-fold-train \
  --repo-id local/openarm-tshirt-fold \
  --chunk-size 60
```

The example `60` is not a default; use the recorded report. Training writes an
OpenArm manifest before launching LeRobot. W&B is disabled. The command stops
if CUDA, the isolated environment, or `/dev/nvidia0` is unavailable.

Teacher-forced validation loss is optional and is not real-robot evidence:

```bash
./openarm-fold report \
  --checkpoint outputs/act-tshirt-fold/checkpoints/last/pretrained_model \
  --validation-root derived/lerobot/openarm-tshirt-fold-validation
```

Physical acceptance remains 8/10 successful folds on familiar shirt types,
5/10 on unseen shirts, and zero safety violations. Running those rollouts is a
separate deployment task.

## Future VLA and world-model data

Do not mix sources silently. Keep demonstrations, failed autonomous rollouts,
and human corrections in provenance-separated dataset versions. After ACT
validates the contract, expand toward 300-500 accepted demonstrations for VLA
fine-tuning. Begin any action-conditioned world model as offline prediction,
not real-robot control.
