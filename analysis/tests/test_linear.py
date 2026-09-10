import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.linear import fit_linear


class LinearTests(unittest.TestCase):
    def row(self, x, y, subset='train'):
        return dict(estimated_ground_distance_m=str(x), target_ground_distance_m=str(y),
                    ground_distance_error_m=str(x-y), subset=subset)

    def test_recovers_affine_map_without_validation_leakage(self):
        rows = [self.row(x, 1.2*x+.4) for x in (1, 3, 8)]
        rows += [self.row(3, 10000, 'validation'), self.row(7, -999, 'test')]
        a, b = fit_linear(rows)
        self.assertAlmostEqual(a, 1.2)
        self.assertAlmostEqual(b, .4)

    def test_rejects_constant_input(self):
        with self.assertRaises(ValueError):
            fit_linear([self.row(1, 2), self.row(1, 3)])
