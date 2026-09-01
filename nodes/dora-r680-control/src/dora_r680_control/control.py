"""Transport-independent Quest-to-R680 command logic.

Data flows through this module as follows::

    Quest axes/buttons
        -> ControllerState
        -> map_controller()
        -> RobotCommand
        -> encode_packet()
        -> UDP bytes

The R680 performs the reverse packet operation with ``decode_packet()`` before
publishing the command to ROS 2. This module imports neither Dora nor ROS 2, so
the mapping and protocol can be tested on any Python 3.10+ machine.

Coordinate and unit conventions follow ``geometry_msgs/msg/Twist``:

* ``linear_x_mps``: positive moves the base forward, in metres per second.
* ``linear_y_mps``: positive moves the base left, in metres per second.
* ``angular_z_radps``: positive rotates counter-clockwise, in radians/second.
* ``elevator_speed``: normalized -1..1 intent; no hardware output exists yet.
"""

from __future__ import annotations

import dataclasses
import json
import math


# Increment this only for an incompatible packet-shape or semantic change. The
# decoder rejects other versions instead of interpreting them incorrectly.
PROTOCOL_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ControllerState:
    """Latest normalized Quest controls before robot-specific scaling.

    Joystick axes are expected in ``[-1.0, 1.0]``. Values outside that range
    are tolerated and clamped by :func:`map_controller`. ``right_y`` is kept in
    the canonical state for future controls but is intentionally unused today.
    """

    # Left thumbstick: sideways and forward/backward base translation.
    left_x: float = 0.0
    left_y: float = 0.0
    # Right thumbstick: X controls base yaw; Y is reserved for future use.
    right_x: float = 0.0
    right_y: float = 0.0
    # Future elevator intent: A=up, B=down, both/neither=stopped.
    button_a: bool = False
    button_b: bool = False


@dataclasses.dataclass(frozen=True)
class MappingConfig:
    """Robot speed limits, axis directions, and joystick deadzone.

    Each scale is the output produced by a fully deflected joystick. Negative
    scales invert an axis without introducing separate inversion flags. The
    default Y and yaw signs translate a stick-right gesture into ROS rightward
    motion/clockwise rotation. Confirm those signs on the real R680 before
    increasing the speed limits.
    """

    # Maximum forward/backward speed in metres per second.
    linear_x_scale: float = 0.2
    # Maximum sideways speed in metres per second; negative inverts stick X.
    linear_y_scale: float = -0.2
    # Maximum yaw speed in radians per second; negative makes stick-right clockwise.
    angular_z_scale: float = -0.5
    # Normalized elevator speed multiplier reserved for the future adapter.
    elevator_scale: float = 1.0
    # Absolute stick values at or below this threshold produce zero motion.
    deadzone: float = 0.15


@dataclasses.dataclass(frozen=True)
class RobotCommand:
    """Canonical robot command independent of UDP, ROS 2, Bluetooth, or CAN.

    Keeping this interface transport-independent lets future adapters reuse the
    same mapping and tests without importing Quest-specific names such as
    ``lsx`` or ``rsx``.
    """

    linear_x_mps: float
    linear_y_mps: float
    angular_z_radps: float
    elevator_speed: float = 0.0

    @classmethod
    def neutral(cls) -> RobotCommand:
        """Return the explicit all-stopped command used during stale input."""
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class ControlPacket:
    """Validated contents of one PC-to-R680 UDP datagram.

    ``sequence`` identifies packet order within one PC process. ``sent_at_ns``
    is the PC wall-clock time for diagnostics; safety timeouts use local
    monotonic clocks instead and therefore do not require synchronized clocks.
    """

    sequence: int
    sent_at_ns: int
    command: RobotCommand


def _shape_axis(value: float, deadzone: float) -> float:
    """Clamp one axis and remove its centre deadzone without a speed jump.

    For a deadzone of 0.15, inputs from -0.15 through +0.15 return zero. The
    remaining magnitude is linearly expanded so that an input of +/-1 still
    returns +/-1. This gives the operator the full configured speed range.

    Raises:
        ValueError: If ``deadzone`` is outside ``[0.0, 1.0)``.
    """
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in the range [0.0, 1.0)")
    value = max(-1.0, min(1.0, float(value)))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    # Remove the unused centre region, then normalize the remaining travel.
    shaped = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(shaped, value)


def map_controller(
    state: ControllerState,
    config: MappingConfig = MappingConfig(),
) -> RobotCommand:
    """Convert one controller snapshot into a canonical robot command.

    Mapping:
        * left Y -> forward/backward velocity
        * left X -> sideways velocity
        * right X -> yaw velocity
        * A/B -> positive/negative elevator intent

    Pressing A and B together yields zero elevator motion, which avoids choosing
    an arbitrary direction for contradictory input.
    """
    # bool converts to 1.0/0.0, so A-B naturally gives +1, -1, or 0.
    elevator_direction = float(state.button_a) - float(state.button_b)
    return RobotCommand(
        linear_x_mps=_shape_axis(state.left_y, config.deadzone)
        * config.linear_x_scale,
        linear_y_mps=_shape_axis(state.left_x, config.deadzone)
        * config.linear_y_scale,
        angular_z_radps=_shape_axis(state.right_x, config.deadzone)
        * config.angular_z_scale,
        elevator_speed=elevator_direction * config.elevator_scale,
    )


def encode_packet(command: RobotCommand, sequence: int, sent_at_ns: int) -> bytes:
    """Encode one command as a compact, versioned UTF-8 JSON datagram.

    The packet keeps base and elevator fields separate so their eventual
    hardware adapters can evolve independently::

        {
          "version": 1,
          "sequence": 42,
          "sent_at_ns": 1234,
          "base": {
            "linear_x_mps": 0.2,
            "linear_y_mps": 0.0,
            "angular_z_radps": -0.3
          },
          "elevator": {"speed": 0.0}
        }

    Raises:
        ValueError: If counters are negative or a command contains NaN/Infinity.
    """
    if sequence < 0 or sent_at_ns < 0:
        raise ValueError("sequence and sent_at_ns must be non-negative")
    values = dataclasses.astuple(command)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("command values must be finite")

    # Explicit field names and physical units make captures readable and avoid
    # relying on a fragile positional array format.
    document = {
        "version": PROTOCOL_VERSION,
        "sequence": sequence,
        "sent_at_ns": sent_at_ns,
        "base": {
            "linear_x_mps": command.linear_x_mps,
            "linear_y_mps": command.linear_y_mps,
            "angular_z_radps": command.angular_z_radps,
        },
        "elevator": {"speed": command.elevator_speed},
    }
    return json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _finite_float(value: object, name: str) -> float:
    """Parse one numeric packet field while rejecting NaN and infinity."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_int(value: object, name: str) -> int:
    """Validate packet counters; booleans are rejected despite being int-like."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def decode_packet(payload: bytes) -> ControlPacket:
    """Validate and decode one untrusted UDP payload.

    Every required field, type, protocol version, and floating-point value is
    checked before a :class:`RobotCommand` can reach ROS 2. Additional unknown
    JSON fields are ignored for forward-compatible diagnostics.

    Raises:
        ValueError: If JSON, version, structure, counters, or values are invalid.
    """
    try:
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("packet must contain a JSON object")
        if document.get("version") != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {document.get('version')}")

        # Separate dictionaries make accidental base/elevator field mixing fail
        # validation rather than silently controlling the wrong mechanism.
        base = document["base"]
        elevator = document["elevator"]
        if not isinstance(base, dict) or not isinstance(elevator, dict):
            raise ValueError("base and elevator must be JSON objects")

        return ControlPacket(
            sequence=_non_negative_int(document["sequence"], "sequence"),
            sent_at_ns=_non_negative_int(document["sent_at_ns"], "sent_at_ns"),
            command=RobotCommand(
                linear_x_mps=_finite_float(base["linear_x_mps"], "linear_x_mps"),
                linear_y_mps=_finite_float(base["linear_y_mps"], "linear_y_mps"),
                angular_z_radps=_finite_float(
                    base["angular_z_radps"], "angular_z_radps"
                ),
                elevator_speed=_finite_float(elevator["speed"], "elevator_speed"),
            ),
        )
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid R680 command packet") from exc
