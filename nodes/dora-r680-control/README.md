# Quest control for the WheelTec R680

This package keeps robot-specific behavior out of the Quest receiver. It has two
entry points:

- `dora-r680-control` runs inside the Dora dataflow on the PC. It maps Quest
  controller inputs to a canonical robot command and sends versioned JSON over
  UDP.
- `r680-control-agent` runs on the R680 computer. It receives those commands and
  publishes `geometry_msgs/msg/Twist` on `/cmd_vel`.

## Architecture and file responsibilities

```text
Quest APK
  -> UDP :5006
  -> existing dora-openarm-quest-receiver
  -> named Dora joystick/button values
  -> pc_node.py
       -> control.py mapping and protocol
       -> UDP :5007 over the R680 Wi-Fi network
  -> r680_agent.py on the R680
  -> ROS 2 geometry_msgs/msg/Twist on /cmd_vel
  -> WheelTec chassis controller
```

- `control.py` owns the transport-independent data models, joystick mapping,
  units, protocol version, JSON encoder, and untrusted-packet validation.
- `pc_node.py` owns Dora event aggregation, Quest packet freshness, command
  timing, and the PC UDP sender.
- `r680_agent.py` owns the R680 UDP listener, ROS `Twist` translation, and
  command-loss stop behavior.
- `tests/test_control.py` protects the shared mapping and protocol interface
  without requiring Dora, network access, or ROS 2.
- `dataflow-r680.yaml` owns the wiring and rates; no existing arm-control
  dataflow is changed.

The mapping defaults are:

| Quest input | R680 command |
| --- | --- |
| Left stick Y | Forward/backward (`linear.x`) |
| Left stick X | Sideways (`linear.y`) |
| Right stick X | Rotation (`angular.z`) |
| A/B | Encoded as elevator up/down; hardware output is intentionally pending |

Signed scale arguments control both speed and direction. Test the signs with
the wheels raised or the base speed limited before operating on the floor.

## UDP command format

Every datagram is one compact UTF-8 JSON object:

```json
{
  "version": 1,
  "sequence": 42,
  "sent_at_ns": 1234,
  "base": {
    "linear_x_mps": 0.2,
    "linear_y_mps": 0.0,
    "angular_z_radps": -0.3
  },
  "elevator": {
    "speed": 0.0
  }
}
```

The R680 rejects malformed JSON, missing fields, unsupported versions,
non-numeric counters, NaN, and infinite command values. The timestamp is for
diagnostics; timeouts use monotonic clocks on each machine and do not require
their wall clocks to be synchronized.

## R680 setup

The WheelTec base must already be running:

```bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

Install this package in a ROS 2 environment that provides `rclpy` and
`geometry_msgs`, then run:

```bash
python3 -m pip install -e nodes/dora-r680-control
r680-control-agent --bind-host 0.0.0.0 --port 5007 --topic /cmd_vel
```

The base installation intentionally does not install Dora or PyArrow on the
R680. Those PC-only dependencies are installed by the dataflow's `pc` extra.

SSH is useful for installing and starting this process, but it is not the
runtime command transport.

## PC setup

Run the dedicated dataflow from the repository root:

```bash
export R680_HOST=<R680_WIFI_IP>
dora build dataflow-r680.yaml --uv
dora run dataflow-r680.yaml --uv
```

PC mapper arguments can be changed in `dataflow-r680.yaml`:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--robot-port` | `5007` | Destination UDP port |
| `--linear-x-scale` | `0.2` | Full-stick forward speed, m/s |
| `--linear-y-scale` | `-0.2` | Full-stick sideways speed, m/s |
| `--angular-z-scale` | `-0.5` | Full-stick yaw speed, rad/s |
| `--deadzone` | `0.15` | Ignored centre range of each stick |
| `--input-timeout` | `0.25` | Quest packet-loss neutral timeout, seconds |

Use a negative scale to reverse an axis. `R680_HOST` is required; `R680_PORT`
can override the default port when `--robot-port` is not explicitly supplied.

R680 agent arguments:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--bind-host` | `0.0.0.0` | Local network interface to listen on |
| `--port` | `5007` | UDP listening port |
| `--topic` | `/cmd_vel` | ROS 2 `Twist` output topic |
| `--command-timeout` | `0.5` | PC packet-loss stop timeout; `0` disables |

The PC sends one UDP packet every 20 ms. The packet contains a protocol version,
sequence number, timestamp, base velocity, and reserved elevator speed. The
PC sends a neutral command when no new Quest UDP packet arrives for 0.25
seconds. The R680 agent independently publishes a zero `Twist` if its command
packets stop for 0.5 seconds.

## Elevator status

A and B are already mapped into the protocol as positive and negative
`elevator.speed`. The R680 agent deliberately does not actuate that value yet:
the elevator's ROS 2 topic/service and physical limits have not been confirmed.
The corresponding `TODO(R680)` in `r680_agent.py` is the only unfinished
hardware connection in this package.
