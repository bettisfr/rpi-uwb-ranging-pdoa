"""Check dependency ordering and static/UAV reconstruction consistency."""

from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from reproduce_paper import ordered_stages
import reproduce_paper
from evaluate_uav_pdoa_position import positions_from_phase
from evaluate_uav_window_gate import windows
from workflow.position_benchmark import reconstruct


class PaperPipelineTests(unittest.TestCase):
    def test_changed_kml_invalidates_cached_pipeline_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'analysis/src'
            source.mkdir(parents=True)
            (root / 'analysis/configs').mkdir()
            (root / 'data').mkdir()
            kml = root / 'data/flight.kml'
            kml.write_text('original trajectory')
            with patch.object(reproduce_paper, 'ROOT', root), patch.object(reproduce_paper, 'SRC', source), patch.object(reproduce_paper, 'ANALYSIS', root / 'analysis'):
                before = reproduce_paper.fingerprint()
                kml.write_text('changed trajectory')
                self.assertNotEqual(before, reproduce_paper.fingerprint())

    def test_uav_dependencies_do_not_require_slow_model_comparisons(self):
        stages = ordered_stages(['uav'])
        self.assertEqual(stages, ['data', 'final-models', 'uav'])

    def test_export_includes_all_required_stages_once(self):
        stages = ordered_stages(['export'])
        self.assertEqual(len(stages), len(set(stages)))
        self.assertLess(stages.index('range-lodo'), stages.index('static-windows'))
        self.assertLess(stages.index('phase-lodo'), stages.index('static-position'))
        self.assertLess(stages.index('final-models'), stages.index('uav'))
        self.assertEqual(stages[-1], 'export')

    def test_uav_zero_yaw_matches_static_forward_reconstruction(self):
        rho = np.array([2., 10., 30.])
        height = np.array([0., 2., 4.])
        phase = np.array([-100., 25., 175.])
        expected_x, expected_y, _ = reconstruct(rho, height, phase)
        x, y, _ = positions_from_phase(rho, height, phase, np.zeros(3), 0.,
            dict(sign=-1., yaw_offset_deg=0.), np.full(3, 5.), np.full(3, -2.))
        np.testing.assert_allclose(x, expected_x + 5., atol=1e-12)
        np.testing.assert_allclose(y, expected_y - 2., atol=1e-12)

    def test_uav_heading_rotates_the_complete_local_position(self):
        rho = np.array([10.])
        height = np.array([2.])
        phase = np.array([-30.])
        local_x, local_y, _ = reconstruct(rho, height, phase)
        x, y, _ = positions_from_phase(rho, height, phase, np.array([90.]), 0.,
            dict(sign=-1., yaw_offset_deg=0.), np.array([0.]), np.array([0.]))
        np.testing.assert_allclose(x, local_y, atol=1e-12)
        np.testing.assert_allclose(y, -local_x, atol=1e-12)

    def test_window_comparison_uses_exactly_its_constituent_observations(self):
        rows = [dict(train_flight='a', flight='b', tag='dw00', seconds=t,
                     tag_x_hat_m=x, tag_y_hat_m=0., tag_x_m=0., tag_y_m=0.,
                     position_error_m=abs(x), pdoa_deg=0., raw_range_m=10., elevation_hat_deg=0.)
                for t, x in [(0., 1.), (.2, -1.), (2., 10.), (3., 10.)]]
        result = windows(rows, 2)
        self.assertEqual(len(result), 1)  # second window has a gap >0.5 s
        self.assertEqual(result[0]['single_sample_mae_m'], 1.)
        self.assertEqual(result[0]['position_error_m'], 0.)


if __name__ == '__main__':
    unittest.main()
