import sys
import unittest
import warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.phase_learning import phase_features
from workflow.distance_models import WeightedDuplicateSVR
from sklearn.multioutput import MultiOutputRegressor


class PhaseFeatureTests(unittest.TestCase):
    def test_capped_fit_keeps_explicit_nonconvergence_status(self):
        X = np.linspace(0, 4, 40).reshape(-1, 1)
        y = np.sin(X[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = WeightedDuplicateSVR(dict(max_iter=1), allow_partial=True).fit(X, y)
            self.assertEqual(model.regressor_.fit_status_, 1)
            with self.assertRaises(RuntimeError):
                WeightedDuplicateSVR(dict(max_iter=1)).fit(X, y)

    def test_two_component_svr_reuses_all_training_samples(self):
        phase = np.linspace(-3, 3, 30)
        X = np.column_stack((np.sin(phase), np.cos(phase)))
        Y = np.column_stack((np.cos(.1*np.sin(phase)), np.sin(.1*np.sin(phase))))
        model = MultiOutputRegressor(WeightedDuplicateSVR(dict(C=1., epsilon=.01)))
        model.fit(np.repeat(X, 2, axis=0), np.repeat(Y, 2, axis=0))
        self.assertEqual(model.predict(X).shape, (30, 2))
        self.assertTrue(all(e.training_samples_ == 60 for e in model.estimators_))
        self.assertTrue(all(e.unique_pairs_ == 30 for e in model.estimators_))

    def test_features_exclude_ground_truth_and_preserve_wrap(self):
        row = dict(pdoa_deg='179', tag_rotation_deg='90', range_m='4', height_difference_m='2',
                   gt_x_m='1', gt_bearing_deg='15', eta_expected_deg='100')
        changed = {**row, 'pdoa_deg': '-181', 'gt_x_m': '999', 'gt_bearing_deg': '999', 'eta_expected_deg': '-100'}
        for enriched in (False, True):
            X, names = phase_features([row], enriched)
            Y, _ = phase_features([changed], enriched)
            np.testing.assert_allclose(X, Y, atol=1e-12)
            self.assertEqual(X.shape[1], 6 if enriched else 2)
            self.assertFalse(any('gt_' in name for name in names))
