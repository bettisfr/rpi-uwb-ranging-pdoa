import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.quadratic import fit_quadratic, predict_quadratic


class QuadraticTests(unittest.TestCase):
    def row(self, x, y, subset='train'):
        return dict(estimated_ground_distance_m=str(x), target_ground_distance_m=str(y),
                    ground_distance_error_m=str(x-y), subset=subset)

    def test_recovers_curve_and_ignores_holdouts(self):
        rows = [self.row(x, .04*x*x+.8*x+.3) for x in (2, 6, 10, 20, 30)]
        model = fit_quadratic(rows)
        self.assertEqual(model, fit_quadratic(rows + [self.row(999, -100, 'validation'),
                                                     self.row(-30, 10000, 'test')]))
        self.assertAlmostEqual(predict_quadratic(15, model), .04*225+.8*15+.3)
        self.assertAlmostEqual(model['quadratic_per_m'], .04)

    def test_rejects_rank_deficient_training(self):
        with self.assertRaises(ValueError):
            fit_quadratic([self.row(x, 2*x) for x in (1, 1, 2, 2)])

    def test_gross_known_geometry_error_is_disabled(self):
        rows = [self.row(x, x) for x in (2, 4, 8, 16)]
        clean = fit_quadratic(rows)
        anomalous = fit_quadratic(rows + [self.row(.36, 30)])
        self.assertAlmostEqual(predict_quadratic(2, clean), predict_quadratic(2, anomalous))
