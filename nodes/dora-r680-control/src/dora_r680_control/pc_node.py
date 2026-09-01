"""PC-side Dora adapter for Quest-to-R680 control.

This executable receives individual Arrow values routed by
``dataflow-r680.yaml``. It caches them as one :class:`ControllerState`, maps that
state at a fixed 50 Hz tick, serializes a versioned command, and sends one UDP
datagram to the R680 agent.

The ``vr_receive_times`` input is the headset heartbeat. The Quest node repeats
its last state on every Dora tick, so joystick event arrival alone cannot prove
that the headset is still connected. When the heartbeat becomes stale, this
node continues transmitting explicit neutral commands instead of stale motion.

Dora outputs:
    ``status``: ``"ready"`` once initialization succeeds.
    ``command``: Exact JSON sent over UDP, useful for logging and inspection.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import time

import dora
import pyarrow as pa

from .control import (
    ControllerState,
    MappingConfig,
    RobotCommand,
    encode_packet,
    map_controller,
)


class UdpCommandSender:
    """Own the PC's datagram socket and fixed R680 destination.

    UDP is intentionally confined to this adapter. Controller mapping and the
    wire format remain reusable if Bluetooth or another transport is added.
    """

    def __init__(self, host: str, port: int) -> None:
        """Create an IPv4 UDP socket targeting ``(host, port)``."""
        self._destination = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: bytes) -> None:
        """Send exactly one encoded command as one UDP datagram."""
        self._socket.sendto(payload, self._destination)

    def close(self) -> None:
        """Release the socket when Dora stops or the node raises an error."""
        self._socket.close()


def _scalar(value) -> object:
    """Extract the Python scalar from a length-one Dora Arrow array."""
    return value[0].as_py()


def _run(args: argparse.Namespace) -> None:
    """Run the Dora event loop until the dataflow stops.

    Input events update cached state; only ``tick`` events transmit. This
    aggregation prevents four joystick inputs from creating four inconsistent
    network packets during one Quest update.
    """
    # External adapters are created once and reused for the process lifetime.
    node = dora.Node()
    sender = UdpCommandSender(args.robot_host, args.robot_port)

    # Signed scales define both maximum velocities and axis directions.
    mapping = MappingConfig(
        linear_x_scale=args.linear_x_scale,
        linear_y_scale=args.linear_y_scale,
        angular_z_scale=args.angular_z_scale,
        elevator_scale=args.elevator_scale,
        deadzone=args.deadzone,
    )
    # Start neutral until every relevant Quest field begins arriving.
    state = ControllerState()
    last_quest_packet: float | None = None
    sequence = 0

    node.send_output("status", pa.array(["ready"]))
    print(f"[r680-control] sending UDP commands to {args.robot_host}:{args.robot_port}")

    try:
        for event in node:
            if event["type"] != "INPUT":
                continue

            event_id = event["id"]
            value = event["value"]

            # Dora delivers each controller channel separately. Replacing one
            # immutable dataclass field produces a coherent cached snapshot.
            if event_id == "joystick_x_left":
                state = dataclasses.replace(state, left_x=float(_scalar(value)))
            elif event_id == "joystick_y_left":
                state = dataclasses.replace(state, left_y=float(_scalar(value)))
            elif event_id == "joystick_x_right":
                state = dataclasses.replace(state, right_x=float(_scalar(value)))
            elif event_id == "joystick_y_right":
                state = dataclasses.replace(state, right_y=float(_scalar(value)))
            elif event_id == "button_a":
                state = dataclasses.replace(state, button_a=bool(_scalar(value)))
            elif event_id == "button_b":
                state = dataclasses.replace(state, button_b=bool(_scalar(value)))
            elif event_id == "vr_receive_times":
                # The Quest receiver republishes its last controller state on
                # each Dora tick. Only this arrival stream proves that a new
                # UDP packet actually came from the headset.
                last_quest_packet = time.monotonic()
            elif event_id == "tick":
                now = time.monotonic()
                # A monotonic clock is immune to wall-clock/NTP adjustments.
                # Continue sending neutral packets so the R680 sees an active,
                # explicitly stopped controller instead of silence.
                command = (
                    RobotCommand.neutral()
                    if last_quest_packet is None
                    or now - last_quest_packet > args.input_timeout
                    else map_controller(state, mapping)
                )
                # Wall-clock nanoseconds are diagnostic metadata only.
                payload = encode_packet(command, sequence, time.time_ns())
                sender.send(payload)

                # Mirror the exact network payload into Dora for recording or
                # debugging without introducing a second command representation.
                node.send_output(
                    "command",
                    pa.array([payload.decode("utf-8")]),
                    {"timestamp": time.time_ns()},
                )
                sequence += 1
                continue
            else:
                continue
    finally:
        # Also runs on exceptions, preventing a leaked file descriptor.
        sender.close()


def main() -> None:
    """Parse PC configuration, validate it, and enter the Dora loop."""
    parser = argparse.ArgumentParser(description="Send Quest commands to an R680")
    parser.add_argument(
        "--robot-host",
        default=os.getenv("R680_HOST"),
        help="R680 Wi-Fi IPv4 address; defaults to the R680_HOST environment variable",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=int(os.getenv("R680_PORT", "5007")),
        help="R680 UDP listener port (default: R680_PORT or 5007)",
    )
    parser.add_argument(
        "--linear-x-scale",
        type=float,
        default=0.2,
        help="full-stick forward speed in m/s; use a negative value to invert",
    )
    parser.add_argument(
        "--linear-y-scale",
        type=float,
        default=-0.2,
        help="full-stick sideways speed in m/s; use a negative value to invert",
    )
    parser.add_argument(
        "--angular-z-scale",
        type=float,
        default=-0.5,
        help="full-stick yaw speed in rad/s; use a negative value to invert",
    )
    parser.add_argument(
        "--elevator-scale",
        type=float,
        default=1.0,
        help="normalized future elevator speed multiplier",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="absolute joystick centre deadzone in [0, 1)",
    )
    parser.add_argument(
        "--input-timeout",
        type=float,
        default=0.25,
        help="seconds without a new Quest UDP packet before sending neutral",
    )
    args = parser.parse_args()
    if not args.robot_host:
        parser.error("--robot-host or R680_HOST is required")
    if not 0 < args.robot_port <= 65535:
        parser.error("--robot-port must be between 1 and 65535")
    if args.input_timeout <= 0:
        parser.error("--input-timeout must be positive")
    if not 0.0 <= args.deadzone < 1.0:
        parser.error("--deadzone must be in the range [0.0, 1.0)")
    _run(args)


if __name__ == "__main__":
    main()
