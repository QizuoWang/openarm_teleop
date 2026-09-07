# OpenArm Teleoperation and Policy Evaluation

VR teleoperation, synchronized dataset collection, review, LeRobot conversion,
and guarded ACT/SmolVLA evaluation for a pedestal-mounted, bimanual
[OpenArm](https://openarm.dev/). Collection uses [Dora](https://dora-rs.ai/);
policy inference runs in a separate process and Python environment.

**Moving to another PC? Start with [PORTING.md](PORTING.md).** It covers fresh
installation, model/SDK transfer, host configuration, and staged startup.

## Capabilities

- Three synchronized RGB views: left wrist, right wrist, and overhead/base.
- Solo VR collection with live preview, explicit episode acceptance/rejection,
  separate session directories, and interrupted-session recovery.
- Browser-based video review and relabeling without deleting source recordings.
- LeRobot exports with explicit joint order, units, gripper semantics, and
  whole-episode training/validation splits.
- Local ACT and SmolVLA inference with camera preview, joint-limit/start-delta
  guards, command slew limiting, physical-motion confirmation, and evaluation
  recording with success/failure labels.

## Safety and support boundary

This repository controls physical hardware. Only one collection/evaluation
process may own the robot and cameras at a time. Keep the workspace clear and
the physical emergency stop immediately accessible. A browser stop button is
not a replacement for a hardware emergency stop.

`preview` reads cameras and joint feedback without enabling motors. Typing
`MOVE` in a motion-enabled mode starts homing; the policy-step limit does not
limit that homing sequence. Never relax joint limits to force a rejected policy
target through. The supplied poses, CAN assignment, gains, and camera serials
are reference configuration for the original rig, not universal calibration.

The portability and policy-adapter changes have **not** been tested on the
destination PC or validated through robot motion. Installing dependencies and
downloading artifacts do not establish hardware readiness.

## Documentation

| Guide | Purpose |
| --- | --- |
| [PC migration](PORTING.md) | Installation, external artifacts, host configuration, first startup |
| [Collection workflow](FOLDING_WORKFLOW.md) | Collect, review, relabel, freeze, and export datasets |
| [Training](TRAINING.md) | Pinned local LeRobot environment and ACT training |
| [ACT deployment](deployments/local_act/README.md) | Legacy radians and new degree-format ACT checkpoints |
| [SmolVLA deployment](deployments/local_smolvla/README.md) | Pinned 30K model, task conditioning, and evaluation |

## Data and model conventions

| Boundary | Joint order | Arm units | Grippers | Cameras |
| --- | --- | --- | --- | --- |
| Dora raw collection | Right, then left | Radians | Logical commands; physical feedback | `wrist_right`, `wrist_left`, `ceiling` |
| LeRobot default export | Left, then right | Degrees | Calibrated degrees, negative-open on both sides | `left_wrist`, `right_wrist`, `base` |
| Policy execution/recording | Right, then left | Radians | Executed physical targets | Original Dora camera names |

Each arm contributes seven joints and one gripper: 16 dimensions in total.
The policy adapter converts observations before normalization and converts
actions after unnormalization. Legacy ACT retains its legacy representation.
Model weights, normalization statistics, and representation metadata must stay
together; renaming camera keys alone is not a valid model conversion.

## Repository and external artifacts

Git contains code, documentation, reference configuration, and a small SmolVLA
deployment manifest. It deliberately excludes datasets, evaluation recordings,
checkpoint weights, Python environments, vendor SDK binaries, and local host
overrides. Clone the source, then transfer the external artifacts described in
[PORTING.md](PORTING.md); a Git clone alone cannot run a trained policy.

## T-shirt folding workflow

The solo VR workflow is exposed through one command:

```bash
./openarm-fold --help
```

It provides `preflight`, `collect`, `review`, `visualize`, `relabel`, `convert`,
`verify-export`, `train`, `report`, and baseline integrity subcommands.
Collection starts with the arms paused, saves lossless raw data to
one directory per session, resumes at the next unused episode, and keeps
failures and interrupted episodes outside the behavioral-cloning split. Dora's
lossless format remains right-first/radian; the default LeRobot 0.6 export is
left-first/degree and uses `left_wrist`, `right_wrist`, and `base` cameras.

See [FOLDING_WORKFLOW.md](FOLDING_WORKFLOW.md) for the executable collection
protocol and [TRAINING.md](TRAINING.md) for the isolated pinned LeRobot/ACT
environment. Policy evaluation uses the separate launchers documented above;
collection preflight never authorizes autonomous motion.

## Configurations

[`metadata.yaml`](metadata.yaml) is metadata used by configurations with real cameras. [`metadata_mujoco.yaml`](metadata_mujoco.yaml) is metadata used by configurations that render cameras with MuJoCo.

### KER configuration

[`dataflow-ker.yaml`](dataflow-ker.yaml) is a configuration for leader-follower teleoperation with real OpenArm units and cameras. A [KER](https://github.com/enactic/dora-openarm-ker) leader arm controls the follower arms while wrist, head and ceiling cameras are recorded.

### VR configuration

[`dataflow-vr.yaml`](dataflow-vr.yaml) is a configuration for VR teleoperation with real OpenArm units and cameras. VR controller poses are received over UDP by [dora-openarm-vr](https://github.com/enactic/dora-openarm-vr), converted to joint positions by inverse kinematics ([dora-openarm-kinematics](https://github.com/enactic/dora-openarm-kinematics)) and sent to the follower arms.

[`dataflow-vr-mujoco.yaml`](dataflow-vr-mujoco.yaml) is the same VR teleoperation but with a MuJoCo simulation ([dora-openarm-mujoco](https://github.com/enactic/dora-openarm-mujoco)) instead of real OpenArm units and cameras. We can use this for testing VR teleoperation and data collection without real hardware.

### WebXR configuration

[`dataflow-webxr-mujoco.yaml`](dataflow-webxr-mujoco.yaml) is a configuration for WebXR teleoperation with a MuJoCo simulation. [dora-openarm-webxr](https://github.com/enactic/dora-openarm-webxr) starts a Web server and the Web browser on a VR device such as Meta Quest 3 or PICO 4 connects to it to stream controller poses. No native VR application is needed.

WebXR requires HTTPS, so a TLS certificate is needed. A self-signed certificate is enough; see the [dora-openarm-webxr setup instructions](https://github.com/enactic/dora-openarm-webxr#setup) for how to generate one. Then run:

```bash
dora build dataflow-webxr-mujoco.yaml --uv
./nodes/dora-openarm-webxr/example/prepare_tls.sh $(hostname).local
dora run dataflow-webxr-mujoco.yaml --uv
```

Open http://localhost:8000/ on the local machine for the data collection UI, and open `https://${YOUR_HOST_NAME}:8443/` in the Web browser on your VR device (where `${HOSTNAME}` matches the value passed to `prepare_tls.sh`) to start teleoperation.

### Dummy configuration

[`dataflow_dummy.yaml`](dataflow_dummy.yaml) is a configuration that doesn't use real OpenArm. We can use this for testing a dataflow without real OpenArm.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
