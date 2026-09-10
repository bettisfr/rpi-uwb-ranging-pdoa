import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.phase import wrap, circular_summary, fit_phase_offset


class PhaseTests(unittest.TestCase):
    def test_wrap_boundary(self):
        self.assertEqual(float(wrap(179-(-179))), -2.)
        self.assertAlmostEqual(circular_summary([179, -179])['mae_deg'], 179.)
        self.assertAlmostEqual(circular_summary([179, -179])['circular_bias_deg'], -180.)

    def test_training_only_and_correction_sign(self):
        offset = fit_phase_offset([-179., 21., 120.], [179., 19., 0.], ['train', 'train', 'validation'])
        self.assertAlmostEqual(offset, 2.)
        self.assertAlmostEqual(float(wrap(-179-offset-179)), 0.)

    def test_undefined_mean_and_empty_training(self):
        self.assertIsNone(circular_summary([0, 180])['circular_bias_deg'])
        with self.assertRaises(ValueError):
            fit_phase_offset([0, 180], [0, 0], ['train', 'train'])
        with self.assertRaises(ValueError):
            fit_phase_offset([0], [0], ['validation'])
