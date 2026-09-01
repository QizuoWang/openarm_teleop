# Data collection configurations for OpenArm with dora-rs

This repository provides data collection configurations for [OpenArm](https://openarm.dev/) with [dora-rs](https://dora-rs.ai/).

## T-shirt folding workflow

The solo VR workflow is exposed through one command:

```bash
./openarm-fold --help
```

It provides `preflight`, `collect`, `review`, `convert`, `train`, and `report`
subcommands. Collection starts with the arms paused, saves lossless raw data to
`folding_data/dataset`, resumes at the next unused episode, and keeps failures
and interrupted episodes outside the initial behavioral-cloning split.

See [FOLDING_WORKFLOW.md](FOLDING_WORKFLOW.md) for the executable collection
protocol and [TRAINING.md](TRAINING.md) for the isolated pinned LeRobot/ACT
environment. Real-robot policy deployment is deliberately not included.

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
