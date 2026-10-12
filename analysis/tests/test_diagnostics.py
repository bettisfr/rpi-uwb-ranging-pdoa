"""Verify the rigid-frame fit used to test firmware coordinate conventions."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workflow.diagnostics import fit_rigid_transform


class RigidTransformTests(unittest.TestCase):
    def test_recovers_reflected_rotation_and_translation(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=(200, 2)) * 10
        angle = np.deg2rad(37.)
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        reflected = rotation @ np.diag([1., -1.])
        target = source @ reflected + np.array([2., -3.])
        fitted, translation = fit_rigid_transform(source, target)
        np.testing.assert_allclose(fitted, reflected, atol=1e-9)
        np.testing.assert_allclose(translation, [2., -3.], atol=1e-9)
        self.assertLess(np.linalg.det(fitted), 0)


if __name__ == "__main__":
    unittest.main()
