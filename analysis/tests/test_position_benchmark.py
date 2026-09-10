import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.position_benchmark import reconstruct


class PositionBenchmarkTests(unittest.TestCase):
    def test_reconstruction_matches_the_adopted_phase_geometry(self):
        rho, h, x = 10., 2., 3.
        r3d = np.hypot(rho, h)
        phase = -180 * x / r3d
        x_hat, y_hat, clipped = reconstruct([rho], [h], [phase])
        self.assertAlmostEqual(x_hat[0], x)
        self.assertAlmostEqual(y_hat[0], np.sqrt(rho*rho - x*x))
        self.assertFalse(clipped[0])

    def test_impossible_phase_is_explicitly_projected(self):
        x_hat, y_hat, clipped = reconstruct([10.], [0.], [-360.])
        self.assertEqual(x_hat[0], 10.)
        self.assertEqual(y_hat[0], 0.)
        self.assertTrue(clipped[0])
