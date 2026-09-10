import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.distance_models import WeightedDuplicateSVR, make_model
from workflow.distance_benchmark import make_folds


class DistanceBenchmarkTests(unittest.TestCase):
    def test_svr_compression_matches_full_sample_objective(self):
        X = np.repeat([[1.], [2.], [4.]], [20, 30, 40], axis=0)
        y = np.repeat([1.5, 2.5, 4.5], [20, 30, 40])
        # Retain a gross outlier with the same input and a different target.
        X = np.vstack((X, [[1.]]))
        y = np.append(y, 20.)
        parameters = dict(C=1., epsilon=.05, gamma=1., tol=1e-7)
        compressed = WeightedDuplicateSVR(parameters).fit(X, y)
        scaler = StandardScaler().fit(X)
        full = SVR(**parameters).fit(scaler.transform(X), y)
        grid = np.linspace(0, 5, 20).reshape(-1, 1)
        np.testing.assert_allclose(compressed.predict(grid), full.predict(scaler.transform(grid)), atol=1e-5)
        self.assertEqual(compressed.training_samples_, 91)
        self.assertEqual(compressed.unique_pairs_, 4)

    def test_lodo_excludes_held_distance_and_original_test(self):
        distance = np.repeat([2., 4., 6.], 4)
        subsets = np.tile(['train', 'validation', 'train', 'test'], 3)
        evaluations = np.zeros(len(distance), dtype=int)
        for _, train, evaluate in make_folds(distance, subsets, 'lodo'):
            self.assertFalse((train & evaluate).any())
            self.assertFalse(((train | evaluate) & (subsets == 'test')).any())
            self.assertFalse(set(distance[train]) & set(distance[evaluate]))
            evaluations += evaluate
        np.testing.assert_array_equal(evaluations, (subsets != 'test').astype(int))

    def test_mlp_scalers_are_fitted_on_supplied_training_only(self):
        config = dict(seed=42, mlp=dict(hidden_layer_sizes=[4], max_iter=2,
                                      early_stopping=False, batch_size=4))
        model = make_model('mlp', config)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit([[1.], [2.], [3.], [4.]], [3., 4., 5., 6.])
        model.predict([[1000.]])
        self.assertEqual(model.regressor_.named_steps['standardscaler'].mean_[0], 2.5)
        self.assertEqual(model.transformer_.mean_[0], 4.5)
