import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.offset import fit_offset, distance_metrics, is_distance_enabled


class OffsetTests(unittest.TestCase):
    def test_fit_uses_only_eligible_training(self):
        rows = [dict(subset='train', ground_distance_error_m='-0.4', quality_enabled=True),
                dict(subset='train', ground_distance_error_m='-0.2', quality_enabled=True),
                dict(subset='train', ground_distance_error_m=''),
                dict(subset='validation', ground_distance_error_m='100'),
                dict(subset='test', ground_distance_error_m='200')]
        self.assertAlmostEqual(fit_offset(rows), -0.3)
        rows[-1]['ground_distance_error_m'] = '-10000'
        self.assertAlmostEqual(fit_offset(rows), -0.3)

    def test_gross_known_geometry_error_is_disabled_but_not_deleted(self):
        self.assertTrue(is_distance_enabled(dict(ground_distance_error_m='-3.0')))
        self.assertFalse(is_distance_enabled(dict(ground_distance_error_m='3.00001')))
        self.assertFalse(is_distance_enabled(dict(ground_distance_error_m='')))

    def test_correction_sign_and_metrics(self):
        truth = [2., 4.]
        raw = [1.7, 3.7]
        metrics = distance_metrics(truth, [value - (-0.3) for value in raw])
        self.assertAlmostEqual(metrics['mae_m'], 0)
        self.assertAlmostEqual(metrics['distance_r2'], 1)
        metrics = distance_metrics(truth, raw)
        self.assertAlmostEqual(metrics['bias_m'], -0.3)
        self.assertAlmostEqual(metrics['rmse_m'], 0.3)
        self.assertIsNone(distance_metrics([2., 2.], [1., 3.])['distance_r2'])

    def test_empty_training_fails(self):
        with self.assertRaises(ValueError):
            fit_offset([dict(subset='test', ground_distance_error_m='1')])
