"""Public interface for the R680 teleoperation module.

The package deliberately exports only transport-independent command models,
mapping, and packet serialization here. The PC Dora process and the R680 ROS 2
process are executable adapters and are not imported automatically. This keeps
the R680-side import usable without installing Dora or PyArrow.
"""

from .control import (
    PROTOCOL_VERSION,
    ControlPacket,
    ControllerState,
    MappingConfig,
    RobotCommand,
    decode_packet,
    encode_packet,
    map_controller,
)

__all__ = [
    # Wire-protocol identifier included in every PC-to-R680 UDP datagram.
    "PROTOCOL_VERSION",
    # Immutable data models used on either side of the UDP connection.
    "ControlPacket",
    "ControllerState",
    "MappingConfig",
    "RobotCommand",
    # The only supported serialization and mapping operations.
    "decode_packet",
    "encode_packet",
    "map_controller",
]
