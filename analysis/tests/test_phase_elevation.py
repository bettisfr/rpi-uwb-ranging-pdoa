"""Verify leave-one-height-out folds and the elevation/range PDoA correction."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workflow.distance_benchmark import make_folds
from workflow.phase import wrap
from workflow.phase_elevation import (estimated_elevation_deg, fit_elevation_range,
                                      gate_mask, predict_elevation_range)


class LeaveOneHeightOutTests(unittest.TestCase):
    def test_folds_hold_out_whole_height_and_exclude_test(self):
        heights = np.repeat([0., 2., 4.], 6)
        subset = np.tile(["train", "validation", "test"], 6)
        folds = list(make_folds(np.zeros(18), subset, "loho", heights))
        self.assertEqual([name for name, _, _ in folds], ["height_0m", "height_2m", "height_4m"])
        for _, train, evaluate in folds:
            self.assertFalse(set(heights[train]) & set(heights[evaluate]))
            self.assertFalse((subset[train | evaluate] == "test").any())
            self.assertFalse((train & evaluate).any())


class ElevationRangeModelTests(unittest.TestCase):
    def test_recovers_elevation_dependent_compression(self):
        rng = np.random.default_rng(3)
        n = 20000
        expected = rng.uniform(-150., 150., n)
        height = rng.choice([0., 2., 4.], n)
        ground = rng.uniform(2., 30., n)
        range_m = np.hypot(ground, height)
        elevation = estimated_elevation_deg(range_m, height)
        # Compression that grows with elevation: a phase-only model cannot fit it.
        scale = 0.9 - 0.4 * np.deg2rad(elevation)
        measured = wrap(scale * expected + rng.normal(0., 2., n))
        model = fit_elevation_range(measured, expected, elevation, range_m)
        corrected, fallbacks = predict_elevation_range(measured, elevation, range_m, model)
        self.assertEqual(fallbacks, 0)
        high = elevation > 30.
        self.assertLess(np.abs(wrap(corrected - expected))[high].mean(),
                        .5 * np.abs(wrap(measured - expected))[high].mean())

    def test_gate_uses_only_observables(self):
        keep = gate_mask(np.array([170., -20., 40.]), np.array([5., 5., 25.]), np.array([0., 10., 50.]),
                         (150., 20., 45.))
        np.testing.assert_array_equal(keep, [False, True, False])


class FinalTestGuardTests(unittest.TestCase):
    def test_refuses_second_evaluation(self):
        import json
        import tempfile
        from workflow.final_test import run_final_test
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "final_test"
            destination.mkdir()
            (destination / "manifest.json").write_text(json.dumps({"status": "complete"}))
            with self.assertRaises(SystemExit):
                run_final_test(Path(directory) / "missing.csv", Path(directory))


if __name__ == "__main__":
    unittest.main()
