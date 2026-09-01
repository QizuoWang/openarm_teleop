"""Behavior tests for the transport-independent R680 module interface.

These tests intentionally avoid Dora, sockets, and ROS 2. They verify the
mapping and wire contract that both deployed processes share.
"""

import json
import unittest

from dora_r680_control.control import (
    ControllerState,
    MappingConfig,
    RobotCommand,
    decode_packet,
    encode_packet,
    map_controller,
)


class ControlTest(unittest.TestCase):
    """Protect controller semantics and packet compatibility across refactors."""

    def test_deadzone_produces_neutral_command(self):
        """Small stick drift must not move any base axis."""
        command = map_controller(
            ControllerState(left_x=0.1, left_y=-0.1, right_x=0.1),
            MappingConfig(deadzone=0.15),
        )
        self.assertEqual(command, RobotCommand.neutral())

    def test_controller_mapping_uses_signed_scales(self):
        """Full deflection must reach configured speed and direction limits."""
        command = map_controller(
            ControllerState(left_x=1.0, left_y=1.0, right_x=1.0, button_a=True),
            MappingConfig(
                linear_x_scale=0.3,
                linear_y_scale=-0.2,
                angular_z_scale=-0.8,
                elevator_scale=0.5,
                deadzone=0.15,
            ),
        )
        self.assertAlmostEqual(command.linear_x_mps, 0.3)
        self.assertAlmostEqual(command.linear_y_mps, -0.2)
        self.assertAlmostEqual(command.angular_z_radps, -0.8)
        self.assertAlmostEqual(command.elevator_speed, 0.5)

    def test_pressing_both_elevator_buttons_cancels_the_command(self):
        """Contradictory elevator inputs must resolve to a stopped mechanism."""
        command = map_controller(ControllerState(button_a=True, button_b=True))
        self.assertEqual(command.elevator_speed, 0.0)

    def test_packet_round_trip(self):
        """Encoding then decoding must preserve metadata and every command field."""
        expected = RobotCommand(0.2, -0.1, 0.5, -1.0)
        packet = decode_packet(encode_packet(expected, sequence=42, sent_at_ns=1234))
        self.assertEqual(packet.sequence, 42)
        self.assertEqual(packet.sent_at_ns, 1234)
        self.assertEqual(packet.command, expected)

    def test_unknown_packet_version_is_rejected(self):
        """An incompatible sender must never reach the ROS 2 output."""
        payload = encode_packet(RobotCommand.neutral(), sequence=0, sent_at_ns=0)
        document = json.loads(payload)
        document["version"] = 999
        with self.assertRaisesRegex(ValueError, "unsupported protocol version"):
            decode_packet(json.dumps(document).encode())


if __name__ == "__main__":
    unittest.main()
