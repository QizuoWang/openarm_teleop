"""Pure checkpoint/representation adapter; never opens robot hardware."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from folding_workflow.representation import (
    LEGACY_REPRESENTATION,
    LEROBOT_OPENARM_REPRESENTATION,
    get_spec,
    inverse_state,
    transform_state,
)


@dataclass(frozen=True)
class CheckpointAdapter:
    representation: str
    chunk_size: int
    policy_type: str = "act"
    task: str | None = None
    policy_python: str | None = None
    vlm_assets: str | None = None

    @property
    def camera_map(self):
        return {
            f"observation.images.{source}": f"observation.images.{target}"
            for source, target in get_spec(self.representation).camera_map.items()
        }

    def model_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,):
            raise ValueError("state must be a 16-D vector")
        return transform_state(state[None, :], self.representation)[0]

    def controller_actions(self, actions):
        # New model grippers are physical degree targets, NOT logical commands.
        # inverse_action would expand them to [-1, 1], changing physical motion.
        # inverse_state instead converts all physical degrees to driver radians
        # and restores the left gripper sign and right-first ordering.
        return inverse_state(actions, self.representation)


def resolve_checkpoint(checkpoint: Path, deployment_manifest: Path | None = None) -> CheckpointAdapter:
    checkpoint = checkpoint.resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    policy_type = config.get("type")
    if policy_type not in ("act", "smolvla"):
        raise ValueError("local deployment supports only ACT and SmolVLA checkpoints")
    deployment = None
    if deployment_manifest is not None:
        deployment = json.loads(deployment_manifest.read_text())
        # Portable manifests resolve paths relative to the repository root.
        # Absolute paths remain supported for existing local manifests.
        for key in ("checkpoint", "policy_python", "vlm_assets"):
            if deployment.get(key):
                path = Path(deployment[key])
                deployment[key] = str((path if path.is_absolute() else ROOT / path).resolve())
        if Path(deployment["checkpoint"]).resolve() != checkpoint:
            raise ValueError("deployment manifest belongs to a different checkpoint")
        if deployment.get("policy_type") != policy_type:
            raise ValueError("deployment manifest policy type conflicts with checkpoint")
        hashes = deployment.get("artifact_sha256", {})
        required = {"config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
        if not required.issubset(hashes):
            raise ValueError("deployment manifest lacks required checkpoint artifact hashes")
        for name, expected in hashes.items():
            artifact = (checkpoint / name).resolve()
            if not artifact.is_relative_to(checkpoint):
                raise ValueError("deployment artifact must be inside checkpoint")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError(f"checkpoint artifact changed since deployment setup: {name}")
    if policy_type == "smolvla":
        if deployment is None:
            raise ValueError("SmolVLA requires an explicit --deployment-manifest")
        if config.get("n_obs_steps") != 1 or config.get("rtc_config") is not None:
            raise ValueError("this SmolVLA adapter requires one observation and RTC disabled")
        if config.get("adapt_to_pi_aloha") or config.get("use_delta_joint_actions_aloha"):
            raise ValueError("ALOHA remapping/delta actions are not supported by this controller")
        if not deployment.get("task", "").strip():
            raise ValueError("SmolVLA deployment requires its training task instruction")
        for key in ("policy_python", "vlm_assets"):
            if not deployment.get(key) or not Path(deployment[key]).exists():
                raise ValueError(f"SmolVLA deployment is missing {key}")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    if inputs.get("observation.state", {}).get("shape") != [16]:
        raise ValueError("checkpoint observation.state must be 16-D")
    if outputs.get("action", {}).get("shape") != [16]:
        raise ValueError("checkpoint action must be 16-D")

    image_keys = {key for key in inputs if key.startswith("observation.images.")}
    matching = []
    for representation in (LEGACY_REPRESENTATION, LEROBOT_OPENARM_REPRESENTATION):
        adapter = CheckpointAdapter(
            representation, int(config["chunk_size"]), policy_type,
            None if deployment is None else deployment.get("task"),
            None if deployment is None else deployment.get("policy_python"),
            None if deployment is None else deployment.get("vlm_assets"),
        )
        if image_keys == set(adapter.camera_map.values()):
            matching.append(adapter)
    if len(matching) != 1:
        raise ValueError("checkpoint has unknown or mixed camera feature names")
    adapter = matching[0]
    for key in image_keys:
        if inputs[key].get("shape") != [3, 360, 640]:
            raise ValueError(f"{key}: deployment expects [3,360,640]")

    # New training writes this manifest at the run root. Do not infer joint
    # units from filename or camera names alone for the new representation.
    manifest = None
    for directory in (checkpoint, *checkpoint.parents):
        if directory == ROOT or directory == directory.parent:
            break
        candidate = directory / "openarm_training_manifest.json"
        if candidate.is_file():
            manifest = json.loads(candidate.read_text())
            break
    declared = None if manifest is None else manifest.get("dataset_representation")
    if deployment is not None:
        deployed_representation = deployment.get("dataset_representation")
        if declared is not None and declared != deployed_representation:
            raise ValueError("deployment and training representations conflict")
        declared = deployed_representation
    if adapter.representation == LEROBOT_OPENARM_REPRESENTATION and declared is None:
        raise ValueError("degree checkpoint requires openarm_training_manifest.json with dataset_representation")
    if declared is not None and declared != adapter.representation:
        raise ValueError("training representation conflicts with checkpoint camera features")
    if adapter.chunk_size < 1:
        raise ValueError("checkpoint chunk_size must be positive")
    return adapter
