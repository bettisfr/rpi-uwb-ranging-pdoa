"""Verify group-level resampling and the circular phase-characteristic fit."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workflow.bootstrap import circular_mean_interval, mean_interval
from workflow.diagnostics import fit_phase_characteristic, invert_characteristic


class GroupedBootstrapTests(unittest.TestCase):
    def test_resamples_configurations_not_samples(self):
        # Two configurations of 1000 identical samples: a sample-level bootstrap
        # would give a tight interval around 0.5, a grouped one spans [0, 1].
        values = np.repeat([0., 1.], 1000)
        groups = np.repeat(["a", "b"], 1000)
        estimate, (low, high), _ = mean_interval(values, groups)
        self.assertEqual(estimate, .5)
        self.assertEqual((low, high), (0., 1.))

    def test_circular_interval_across_wrap(self):
        angles = np.array([179., -179., 178., -178.] * 50)
        groups = np.arange(len(angles)).astype(str)
        estimate, (low, high), _ = circular_mean_interval(angles, groups)
        self.assertAlmostEqual(abs(estimate), 180., places=6)
        self.assertLess(high - low, 2.)


class PhaseCharacteristicTests(unittest.TestCase):
    def test_recovers_scale_and_offset(self):
        rng = np.random.default_rng(1)
        expected = rng.uniform(-120., 120., 5000)
        measured = (0.9 * expected + 10. + rng.normal(0., 3., expected.size) + 180.) % 360. - 180.
        groups = (np.arange(expected.size) // 50).astype(str)
        model = fit_phase_characteristic(measured, expected, groups, n_boot=200)
        self.assertAlmostEqual(model["scale"], .9, delta=.01)
        self.assertAlmostEqual(model["offset_deg"], 10., delta=.5)
        self.assertLessEqual(model["scale_ci_low"], .9 + 1e-9)
        self.assertGreaterEqual(model["scale_ci_high"], .9 - 1e-9)
        error = invert_characteristic(measured, model) - expected
        self.assertLess(np.abs(error).mean(), 5.)

    def test_inverse_saturates_at_endfire_instead_of_flipping_side(self):
        model = dict(scale=.9, offset_deg=0.)
        np.testing.assert_allclose(invert_characteristic(np.array([170., -170.]), model), [180., -180.])


if __name__ == "__main__":
    unittest.main()
