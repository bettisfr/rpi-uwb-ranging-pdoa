"""Verify change-point RTK interpolation of 1 Hz fixes held in 10 Hz rows."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_uav_range import change_points, track_at


class ChangePointTrackTests(unittest.TestCase):
    def setUp(self):
        # Node moving at 2 m/s along x; fix refreshed every 1 s, rows every 0.1 s.
        rows_t = np.round(np.arange(0., 5.01, .1), 1)
        held_x = 2. * np.floor(rows_t)
        changed = change_points(rows_t, held_x)
        self.track = dict(t_xy=rows_t[changed], xy=np.column_stack((held_x[changed], np.zeros(changed.sum()))),
                          t_h=np.array([0., 5.]), h=np.array([3., 3.]))
        self.changed = changed

    def test_detects_only_refreshes(self):
        self.assertEqual(int(self.changed.sum()), 6)

    def test_interpolates_between_fixes_with_speed(self):
        x, _, height, gap, speed, valid = track_at(self.track, np.array([1.5, 2.25]))
        np.testing.assert_allclose(x, [3., 4.5])
        np.testing.assert_allclose(speed, [2., 2.])
        np.testing.assert_allclose(gap, [1., 1.])
        np.testing.assert_allclose(height, [3., 3.])
        self.assertTrue(valid.all())

    def test_rejects_missed_update_and_out_of_range(self):
        track = dict(self.track, t_xy=np.array([0., 1., 3.]), xy=np.array([[0., 0.], [2., 0.], [6., 0.]]))
        _, _, _, _, _, valid = track_at(track, np.array([.5, 2., 4.]))
        np.testing.assert_array_equal(valid, [True, False, False])


if __name__ == "__main__":
    unittest.main()
