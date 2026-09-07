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

This writes `.openarm-fold/session.yaml`. Each new session receives its own raw
dataset at `folding_data/sessions/<session-id>/dataset`; sessions never append
episodes to one another. Garment metadata is selected once per batch, not once
per episode.

The earlier shared dataset remains at `folding_data/dataset` and is not moved.
Pass `--raw-root folding_data/dataset` when reviewing or converting that legacy
smoke data.

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

The monitor shows live ceiling, left-wrist, and right-wrist previews at 10 FPS
alongside the full-rate stream-health measurements. Preview requests are
cache-disabled and retain only the newest frame in memory.

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
./openarm-fold visualize --episode 0 --open
./openarm-fold review
./openarm-fold report
```

`review` builds a synchronized three-camera player for every episode and links
each player from the batch page. Stop the collector before correcting a label:

```bash
./openarm-fold review --serve --open
```

The interactive review tool binds only to `127.0.0.1:8001`, requires a
per-process form token and confirmation for each label change, and refuses to
relabel while the collection UI is running on port 8000. Press Ctrl+C to stop
the review server.

```bash
./openarm-fold relabel \
  --episode 12 \
  --status failed \
  --failure-reason bad_demonstration

./openarm-fold relabel --episode 12 --status accepted
```

Relabeling updates the per-episode manifest and dataset metadata while preserving
all raw camera and robot files.

Without `--raw-root`, these commands use the dataset recorded in the active
`.openarm-fold/session.yaml`. To inspect another session, pass its explicit
root, for example `--raw-root folding_data/sessions/fold-pilot-shirt02/dataset`.

Outputs:

- `reports/folding/episode-<id>-player.html`: synchronized wrist-left, ceiling,
  and wrist-right playback. Space pauses; Left/Right seek one second; the speed
  selector supports 0.25x to 2x playback.
- `reports/folding/episodes.html`: first/middle/final ceiling frames with links
  to every synchronized episode player.
- `reports/folding/dataset-report.json`: complete-episode split, garment
  counts, failure reasons, duration distribution, batch gate, and the ACT chunk
  candidate after 25 accepted episodes.

Pause collection when rejection exceeds 10 percent, a required stream fails,
or the canonical fold drifts. Duration is only a warning; the first batch's
95th percentile becomes the soft threshold.

## 5. Convert complete episodes to LeRobotDataset v3

Freeze the reviewed raw data and the working ACT baseline first. This creates a
compact hash manifest and an application-level marker; it does not duplicate
the large artifacts. After this command, collection resume and relabeling are
refused for the frozen raw dataset.

```bash
./openarm-fold freeze-baseline \
  --raw-root folding_data/sessions/fold-train-shirt01-batch01/dataset

./openarm-fold verify-baseline
```

Set up the pinned checkout and environment in [TRAINING.md](TRAINING.md), then
export the train and validation splits to new directories:

```bash
./openarm-fold convert \
  --raw-root folding_data/sessions/fold-train-shirt01-batch01/dataset \
  --dataset-split train \
  --output-root derived/lerobot06/openarm-tshirt-fold-train

./openarm-fold convert \
  --raw-root folding_data/sessions/fold-train-shirt01-batch01/dataset \
  --dataset-split validation \
  --output-root derived/lerobot06/openarm-tshirt-fold-validation
```

Conversion uses a fixed 30 Hz overlapping timeline:

- nearest original camera frame within 30 ms;
- linearly interpolated measured joint state;
- most recent command at or before the frame timestamp;
- no fabricated camera, state, or action samples.

The default `lerobot-openarm-v1` representation has 16-D left-arm-then-right-arm
state and action in degrees. Both gripper action targets use the same
negative-open convention with a conservative range of `[-22.9183, 0]` degrees;
measured state remains unclipped. Raw logical gripper actions are validated
before conversion. The model-facing schema contains
only state, action, task, and the `left_wrist`, `right_wrist`, and `base` camera
streams; velocity and effort remain in the lossless raw source.

Raw 640 x 360 JPEGs are retained. Derived camera streams are H.264 MP4. The
converter refuses to overwrite an existing output, stages incomplete work away
from the final path, recomputes statistics, and writes an alignment manifest.
The old right-first/radian behavior remains available only when explicitly
requested with `--representation dora-rad-v1` and a different output path.

Verify each completed export against every transformed raw state/action row and
decode beginning, middle, and ending frames from every camera:

```bash
./openarm-fold verify-export \
  --raw-root folding_data/sessions/fold-train-shirt01-batch01/dataset \
  --dataset-root derived/lerobot06/openarm-tshirt-fold-train \
  --dataset-split train \
  --expected-episodes 52 \
  --expected-frames 91589

./openarm-fold verify-export \
  --raw-root folding_data/sessions/fold-train-shirt01-batch01/dataset \
  --dataset-root derived/lerobot06/openarm-tshirt-fold-validation \
  --dataset-split validation \
  --expected-episodes 6 \
  --expected-frames 10299
```

## 6. Train and report

Copy the chunk candidate from the 25-episode report into the command or the
pinned profile:

```bash
./openarm-fold preflight --training

./openarm-fold train \
  --dataset-root derived/lerobot06/openarm-tshirt-fold-train \
  --chunk-size 60
```

The example `60` is not a default; use the recorded report. The repository ID is
read from the conversion manifest. Training writes an OpenArm manifest before
launching the pinned local LeRobot checkout. W&B and Hub upload are disabled.
The command stops if the dataset representation, local revision, clean checkout,
CUDA environment, or `/dev/nvidia0` does not match the pinned profile.

Teacher-forced validation loss is optional and is not real-robot evidence:

```bash
./openarm-fold report \
  --checkpoint outputs/act-tshirt-fold/checkpoints/last/pretrained_model \
  --validation-root derived/lerobot06/openarm-tshirt-fold-validation
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
