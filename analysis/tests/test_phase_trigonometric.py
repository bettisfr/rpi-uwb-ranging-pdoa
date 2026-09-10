import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.phase import wrap
from workflow.phase_trigonometric import fit_trigonometric, predict_correction


class TrigonometricTests(unittest.TestCase):
    def test_periodic_error_and_holdout_isolation(self):
        phase = np.linspace(-179, 179, 100)
        error = 10*np.sin(np.deg2rad(phase))
        expected = wrap(phase-error)
        subsets = np.array(['train']*80+['validation']*20)
        model = fit_trigonometric(phase, expected, subsets)
        changed = expected.copy()
        changed[80:] = 0
        self.assertEqual(model, fit_trigonometric(phase, changed, subsets))
        correction, _, _ = predict_correction(phase, model)
        self.assertLess(np.mean(np.abs(wrap(correction-error))), .1)
        shifted, _, _ = predict_correction(phase+360, model)
        np.testing.assert_allclose(correction, shifted, atol=1e-10)

    def test_degenerate_vector_falls_back(self):
        model = dict(harmonics=2, coefficients=np.zeros((5, 2)).tolist(),
                     minimum_vector_norm=1e-8, fallback_offset_deg=3.)
        correction, _, fallback = predict_correction([0, 180], model)
        np.testing.assert_array_equal(correction, [3, 3])
        self.assertTrue(fallback.all())

    def test_rank_deficiency(self):
        with self.assertRaises(ValueError):
            fit_trigonometric([0]*10, [0]*10, ['train']*10)
