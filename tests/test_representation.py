import unittest

import numpy as np

from folding_workflow.representation import (
    LEGACY_REPRESENTATION,
    LEROBOT_OPENARM_REPRESENTATION,
    SAFE_GRIPPER_OPEN_DEG,
    inverse_action,
    inverse_state,
    transform_action,
    transform_state,
)


class RepresentationTest(unittest.TestCase):
    def test_state_is_left_first_degrees_with_canonical_gripper_sign(self):
        source = np.arange(16, dtype=np.float32)[None, :] / 10
        converted = transform_state(source, LEROBOT_OPENARM_REPRESENTATION)
        expected_left = np.rad2deg(source[:, 8:]).astype(np.float32)
        expected_left[:, 7] *= -1
        expected_right = np.rad2deg(source[:, :8]).astype(np.float32)
        np.testing.assert_allclose(converted, np.concatenate([expected_left, expected_right], axis=1))

    def test_state_round_trip(self):
        source = np.linspace(-1, 1, 48, dtype=np.float32).reshape(3, 16)
        np.testing.assert_allclose(
            inverse_state(transform_state(source, LEROBOT_OPENARM_REPRESENTATION), LEROBOT_OPENARM_REPRESENTATION),
            source,
            atol=1e-6,
        )

    def test_action_gripper_endpoints_and_round_trip(self):
        source = np.zeros((2, 16), dtype=np.float32)
        source[0, 7], source[0, 15] = -1, 1
        source[1, 7], source[1, 15] = 0, 0
        converted = transform_action(source, LEROBOT_OPENARM_REPRESENTATION)
        np.testing.assert_allclose(converted[:, 7], [-SAFE_GRIPPER_OPEN_DEG, 0])
        np.testing.assert_allclose(converted[:, 15], [-SAFE_GRIPPER_OPEN_DEG, 0])
        np.testing.assert_allclose(
            inverse_action(converted, LEROBOT_OPENARM_REPRESENTATION), source, atol=1e-6
        )

    def test_action_rejects_unknown_gripper_domains(self):
        source = np.zeros((1, 16), dtype=np.float32)
        source[0, 7] = 0.2
        with self.assertRaisesRegex(ValueError, "right gripper"):
            transform_action(source, LEROBOT_OPENARM_REPRESENTATION)

    def test_legacy_representation_is_identity(self):
        source = np.linspace(-1, 1, 32, dtype=np.float32).reshape(2, 16)
        np.testing.assert_array_equal(transform_state(source, LEGACY_REPRESENTATION), source)
        np.testing.assert_array_equal(transform_action(source, LEGACY_REPRESENTATION), source)


if __name__ == "__main__":
    unittest.main()
