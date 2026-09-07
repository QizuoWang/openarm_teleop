"""Model-facing representations for lossless OpenArm recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_REPRESENTATION = "dora-rad-v1"
LEROBOT_OPENARM_REPRESENTATION = "lerobot-openarm-v1"
REPRESENTATIONS = (LEROBOT_OPENARM_REPRESENTATION, LEGACY_REPRESENTATION)

SAFE_GRIPPER_OPEN_RAD = 0.4
SAFE_GRIPPER_OPEN_DEG = float(np.degrees(SAFE_GRIPPER_OPEN_RAD))
GRIPPER_DOMAIN_TOLERANCE = 1e-4

LEGACY_JOINT_NAMES = tuple(
    [*(f"right_joint_{index}" for index in range(1, 8)), "right_gripper"]
    + [*(f"left_joint_{index}" for index in range(1, 8)), "left_gripper"]
)
LEROBOT_JOINT_NAMES = tuple(
    [*(f"left_joint_{index}.pos" for index in range(1, 8)), "left_gripper.pos"]
    + [*(f"right_joint_{index}.pos" for index in range(1, 8)), "right_gripper.pos"]
)

LEGACY_CAMERA_MAP = {
    "wrist_right": "wrist_right",
    "wrist_left": "wrist_left",
    "ceiling": "ceiling",
}
LEROBOT_CAMERA_MAP = {
    "wrist_left": "left_wrist",
    "wrist_right": "right_wrist",
    "ceiling": "base",
}


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    joint_names: tuple[str, ...]
    camera_map: dict[str, str]
    robot_type: str
    joint_order: str
    joint_units: str
    gripper_units: str
    include_telemetry: bool


SPECS = {
    LEGACY_REPRESENTATION: RepresentationSpec(
        name=LEGACY_REPRESENTATION,
        joint_names=LEGACY_JOINT_NAMES,
        camera_map=LEGACY_CAMERA_MAP,
        robot_type="openarm_bimanual",
        joint_order="right_then_left",
        joint_units="radians",
        gripper_units="source_logical_action_and_radian_state",
        include_telemetry=True,
    ),
    LEROBOT_OPENARM_REPRESENTATION: RepresentationSpec(
        name=LEROBOT_OPENARM_REPRESENTATION,
        joint_names=LEROBOT_JOINT_NAMES,
        camera_map=LEROBOT_CAMERA_MAP,
        robot_type="bi_openarm_follower",
        joint_order="left_then_right",
        joint_units="degrees",
        gripper_units="safe_calibrated_degrees",
        include_telemetry=False,
    ),
}


def get_spec(name: str) -> RepresentationSpec:
    try:
        return SPECS[name]
    except KeyError as error:
        raise ValueError(f"unsupported representation: {name}") from error


def _require_matrix(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 16:
        raise ValueError(f"{label} must have shape (frames, 16), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def _split_source(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values[:, :8].copy(), values[:, 8:].copy()


def _validate_logical_gripper(values: np.ndarray, low: float, high: float, side: str) -> None:
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < low - GRIPPER_DOMAIN_TOLERANCE or maximum > high + GRIPPER_DOMAIN_TOLERANCE:
        raise ValueError(
            f"{side} gripper action outside source domain [{low}, {high}]: "
            f"observed [{minimum}, {maximum}]"
        )


def transform_state(values: np.ndarray, representation: str) -> np.ndarray:
    """Transform right-first source qpos into the requested model representation."""
    source = _require_matrix(values, "state")
    if representation == LEGACY_REPRESENTATION:
        return source.copy()
    get_spec(representation)
    right, left = _split_source(source)
    right = np.rad2deg(right).astype(np.float32)
    left = np.rad2deg(left).astype(np.float32)
    # Dora's left gripper opens in the positive direction. Native LeRobot
    # OpenArm data uses the same negative-open convention for both grippers.
    left[:, 7] *= -1.0
    return np.concatenate([left, right], axis=1).astype(np.float32)


def inverse_state(values: np.ndarray, representation: str) -> np.ndarray:
    """Return a model-facing state to the Dora right-first/radian convention."""
    target = _require_matrix(values, "state")
    if representation == LEGACY_REPRESENTATION:
        return target.copy()
    get_spec(representation)
    left, right = target[:, :8].copy(), target[:, 8:].copy()
    left[:, 7] *= -1.0
    left = np.deg2rad(left).astype(np.float32)
    right = np.deg2rad(right).astype(np.float32)
    return np.concatenate([right, left], axis=1).astype(np.float32)


def transform_action(values: np.ndarray, representation: str) -> np.ndarray:
    """Transform arm radians and logical grippers into native LeRobot degrees."""
    source = _require_matrix(values, "action")
    if representation == LEGACY_REPRESENTATION:
        return source.copy()
    get_spec(representation)
    right, left = _split_source(source)
    _validate_logical_gripper(right[:, 7], -1.0, 0.0, "right")
    _validate_logical_gripper(left[:, 7], 0.0, 1.0, "left")
    right[:, :7] = np.rad2deg(right[:, :7])
    left[:, :7] = np.rad2deg(left[:, :7])
    right[:, 7] = np.clip(right[:, 7], -1.0, 0.0) * SAFE_GRIPPER_OPEN_DEG
    left[:, 7] = -np.clip(left[:, 7], 0.0, 1.0) * SAFE_GRIPPER_OPEN_DEG
    return np.concatenate([left, right], axis=1).astype(np.float32)


def inverse_action(values: np.ndarray, representation: str) -> np.ndarray:
    """Return model-facing actions to Dora arm radians and logical grippers."""
    target = _require_matrix(values, "action")
    if representation == LEGACY_REPRESENTATION:
        return target.copy()
    get_spec(representation)
    left, right = target[:, :8].copy(), target[:, 8:].copy()
    left[:, :7] = np.deg2rad(left[:, :7])
    right[:, :7] = np.deg2rad(right[:, :7])
    left[:, 7] = -left[:, 7] / SAFE_GRIPPER_OPEN_DEG
    right[:, 7] = right[:, 7] / SAFE_GRIPPER_OPEN_DEG
    return np.concatenate([right, left], axis=1).astype(np.float32)

