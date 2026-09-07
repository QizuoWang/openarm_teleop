"""Offline adapter tests; no hardware imports or robot access."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deployments.local_act.representation_adapter import CheckpointAdapter, resolve_checkpoint
from folding_workflow.representation import (
    LEGACY_REPRESENTATION,
    LEROBOT_OPENARM_REPRESENTATION,
    SAFE_GRIPPER_OPEN_DEG,
)


class DeploymentRepresentationTest(unittest.TestCase):
    def test_smolvla_requires_pinned_manifest_and_absolute_actions(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 50, "smolvla")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "pretrained_model"
            checkpoint.mkdir()
            config = {
                "type": "smolvla", "chunk_size": 50, "n_obs_steps": 1,
                "rtc_config": None, "adapt_to_pi_aloha": False,
                "use_delta_joint_actions_aloha": False,
                "input_features": {
                    "observation.state": {"shape": [16]},
                    **{key: {"shape": [3, 360, 640]} for key in adapter.camera_map.values()},
                },
                "output_features": {"action": {"shape": [16]}},
            }
            config_path = checkpoint / "config.json"
            config_path.write_text(json.dumps(config))
            names = ["config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"]
            for name in names[1:]:
                (checkpoint / name).write_text("{}")
            with self.assertRaisesRegex(ValueError, "explicit --deployment-manifest"):
                resolve_checkpoint(checkpoint)
            manifest = {
                "policy_type": "smolvla", "checkpoint": str(checkpoint),
                "dataset_representation": LEROBOT_OPENARM_REPRESENTATION,
                "task": "Fold the T-shirt into a compact rectangle.",
                "policy_python": str(config_path), "vlm_assets": str(root),
                "artifact_sha256": {
                    name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                    for name in names
                },
            }
            manifest_path = root / "deployment.json"
            manifest_path.write_text(json.dumps(manifest))
            resolved = resolve_checkpoint(checkpoint, manifest_path)
            self.assertEqual(resolved.policy_type, "smolvla")
            self.assertEqual(resolved.chunk_size, 50)
            self.assertEqual(resolved.task, manifest["task"])
            config["use_delta_joint_actions_aloha"] = True
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                resolve_checkpoint(checkpoint, manifest_path)
            manifest["artifact_sha256"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "delta actions"):
                resolve_checkpoint(checkpoint, manifest_path)

    def test_legacy_is_unchanged(self):
        adapter = CheckpointAdapter(LEGACY_REPRESENTATION, 85)
        values = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
        np.testing.assert_array_equal(adapter.model_state(values), values)
        np.testing.assert_array_equal(adapter.controller_actions(values[None]), values[None])
        self.assertTrue(all(source == target for source, target in adapter.camera_map.items()))

    def test_measured_state_order_units_and_grippers(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 85)
        state = np.zeros(16, dtype=np.float32)
        state[0], state[8] = np.pi / 2, -np.pi / 4
        state[7], state[15] = -0.2, 0.1
        actual = adapter.model_state(state)
        np.testing.assert_allclose(actual[[0, 8]], [-45, 90], atol=1e-5)
        np.testing.assert_allclose(actual[[7, 15]], np.rad2deg([-0.1, -0.2]), atol=1e-5)

    def test_degree_actions_produce_physical_not_logical_grippers(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 85)
        actions = np.zeros((3, 16), dtype=np.float32)
        actions[:, 0], actions[:, 8] = -45, 90
        actions[:, 7] = [0, -SAFE_GRIPPER_OPEN_DEG / 2, -SAFE_GRIPPER_OPEN_DEG]
        actions[:, 15] = actions[:, 7]
        actual = adapter.controller_actions(actions)
        np.testing.assert_allclose(actual[:, 0], np.pi / 2, atol=1e-6)
        np.testing.assert_allclose(actual[:, 8], -np.pi / 4, atol=1e-6)
        np.testing.assert_allclose(actual[:, 7], [0, -0.2, -0.4], atol=1e-6)
        np.testing.assert_allclose(actual[:, 15], [0, 0.2, 0.4], atol=1e-6)

    def test_camera_mapping(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 85)
        self.assertEqual(adapter.camera_map, {
            "observation.images.wrist_left": "observation.images.left_wrist",
            "observation.images.wrist_right": "observation.images.right_wrist",
            "observation.images.ceiling": "observation.images.base",
        })

    def test_nonfinite_rejected(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 85)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                adapter.model_state(np.full(16, value))
            with self.assertRaises(ValueError):
                adapter.controller_actions(np.full((1, 16), value))

    def test_new_checkpoint_requires_matching_manifest(self):
        adapter = CheckpointAdapter(LEROBOT_OPENARM_REPRESENTATION, 85)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "100000" / "pretrained_model"
            checkpoint.mkdir(parents=True)
            config = {
                "type": "act", "chunk_size": 85,
                "input_features": {
                    "observation.state": {"shape": [16]},
                    **{key: {"shape": [3, 360, 640]} for key in adapter.camera_map.values()},
                },
                "output_features": {"action": {"shape": [16]}},
            }
            (checkpoint / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "requires openarm_training_manifest"):
                resolve_checkpoint(checkpoint)
            manifest = root / "openarm_training_manifest.json"
            manifest.write_text(json.dumps({"dataset_representation": LEGACY_REPRESENTATION}))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve_checkpoint(checkpoint)
            manifest.write_text(json.dumps({"dataset_representation": LEROBOT_OPENARM_REPRESENTATION}))
            self.assertEqual(resolve_checkpoint(checkpoint), adapter)
            config["input_features"]["observation.images.ceiling"] = {"shape": [3, 360, 640]}
            (checkpoint / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "mixed camera"):
                resolve_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
