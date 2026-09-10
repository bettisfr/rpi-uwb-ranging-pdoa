import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.forest import fit_forest


class ForestTests(unittest.TestCase):
    def test_training_only_reproducibility_and_serialization(self):
        rows = [dict(subset='train', estimated_ground_distance_m=str(x-.35),
                     target_ground_distance_m=str(x), ground_distance_error_m='-.35')
                for x in (2, 4, 6) for _ in range(50)]
        model = fit_forest(rows)
        other = fit_forest(rows + [dict(subset='validation', ground_distance_error_m='0',
                                       estimated_ground_distance_m='1', target_ground_distance_m='9999'),
                                  dict(subset='test', ground_distance_error_m='0',
                                       estimated_ground_distance_m='3', target_ground_distance_m='9999')])
        x = [[1.65], [3.65], [5.65]]
        np.testing.assert_allclose(model.predict(x), other.predict(x))
        self.assertEqual(model.n_features_in_, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(model.predict(x), joblib.load(path).predict(x))

    def test_no_training_fails(self):
        with self.assertRaises(ValueError):
            fit_forest([])
