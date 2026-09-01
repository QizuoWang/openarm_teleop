"""R680-side UDP-to-ROS 2 adapter.

Run this process on the R680 computer after WheelTec's chassis launch file. A
non-blocking UDP socket receives versioned commands from the PC while a 50 Hz
ROS timer drains queued datagrams and publishes only the freshest valid command
as ``geometry_msgs/msg/Twist`` on ``/cmd_vel``.

The agent intentionally does not import Dora. Its installation uses the R680's
existing ROS 2 Python environment. A local monotonic timeout publishes one
explicit zero ``Twist`` if the PC connection stops.
"""

from __future__ import annotations

import argparse
import socket
import time

from .control import RobotCommand, decode_packet


def main() -> None:
    """Parse ROS/UDP configuration and run the bridge until interrupted."""
    parser = argparse.ArgumentParser(description="Bridge R680 UDP commands to ROS 2")
    parser.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="local IPv4 interface to listen on (default: every interface)",
    )
    parser.add_argument("--port", type=int, default=5007, help="UDP listen port")
    parser.add_argument(
        "--topic", default="/cmd_vel", help="ROS 2 Twist topic for the R680 base"
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=0.5,
        help="seconds without UDP before publishing stop; 0 disables the timeout",
    )
    args, ros_args = parser.parse_known_args()
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.command_timeout < 0:
        parser.error("--command-timeout cannot be negative")

    # ROS imports stay inside the entry point so protocol tests and the base
    # package remain importable on a PC without ROS 2 installed.
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    class R680ControlAgent(Node):
        """Own the UDP listener, ROS publisher, and connection timeout state."""

        def __init__(self) -> None:
            """Bind UDP, create the Twist publisher, and start a 50 Hz poll."""
            super().__init__("r680_quest_control")
            self._twist_type = Twist
            self._publisher = self.create_publisher(Twist, args.topic, 10)

            # Non-blocking I/O keeps all work in the ROS executor thread and
            # avoids a second thread or synchronization around the publisher.
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((args.bind_host, args.port))
            self._socket.setblocking(False)
            self._last_receive: float | None = None
            self._stopped_for_timeout = False
            self._timer = self.create_timer(0.02, self._poll)
            self.get_logger().info(
                f"listening on UDP {args.bind_host}:{args.port}; publishing {args.topic}"
            )

        def _publish(self, command: RobotCommand) -> None:
            """Translate canonical SI-unit base fields into one ROS Twist."""
            twist = self._twist_type()
            twist.linear.x = command.linear_x_mps
            twist.linear.y = command.linear_y_mps
            twist.angular.z = command.angular_z_radps
            self._publisher.publish(twist)

            # TODO(R680): Connect command.elevator_speed after the R680 elevator
            # ROS 2 topic or service has been identified and documented.

        def _poll(self) -> None:
            """Drain queued UDP packets, publish the freshest, or stop on timeout."""
            latest = None

            # UDP can accumulate several packets between timer callbacks. Old
            # teleoperation commands are useless, so retain only the newest
            # valid packet and minimize control latency.
            while True:
                try:
                    payload, _source = self._socket.recvfrom(4096)
                except BlockingIOError:
                    break
                try:
                    # decode_packet validates untrusted network input before any
                    # value can be copied into a motor command.
                    latest = decode_packet(payload)
                except ValueError as exc:
                    self.get_logger().warning(f"ignored invalid command: {exc}")

            if latest is not None:
                self._publish(latest.command)
                self._last_receive = time.monotonic()
                self._stopped_for_timeout = False
                return

            # Publish the stop only once per outage. Repeated zero messages are
            # unnecessary because WheelTec already consumes the last Twist.
            if (
                args.command_timeout > 0
                and self._last_receive is not None
                and time.monotonic() - self._last_receive > args.command_timeout
                and not self._stopped_for_timeout
            ):
                self._publish(RobotCommand.neutral())
                self._stopped_for_timeout = True
                self.get_logger().warning("command timeout; published a stop command")

        def close(self) -> None:
            """Publish a final stop and release the UDP socket during shutdown."""
            self._publish(RobotCommand.neutral())
            self._socket.close()

    # Unknown arguments from argparse are preserved for ROS remapping/options.
    rclpy.init(args=ros_args)
    node = R680ControlAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
