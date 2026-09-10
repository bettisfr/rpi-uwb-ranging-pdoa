import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from workflow.phase_stratified import strategy_key


class StratifiedPhaseTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(tag='dw00', tag_rotation_deg='0'),
            dict(tag='dw00', tag_rotation_deg='180'),
            dict(tag='dw01', tag_rotation_deg='0'),
        ]

    def test_deployment_strata_use_only_known_tag_and_rotation(self):
        self.assertEqual(strategy_key(self.rows, 'global').tolist(), ['global'] * 3)
        self.assertEqual(strategy_key(self.rows, 'bearing').tolist(), ['dw00', 'dw00', 'dw01'])
        self.assertEqual(strategy_key(self.rows, 'rotation').tolist(), ['0', '180', '0'])
        self.assertEqual(strategy_key(self.rows, 'bearing_rotation').tolist(),
                         ['dw00|0', 'dw00|180', 'dw01|0'])

    def test_unknown_strategy_fails(self):
        with self.assertRaises(ValueError):
            strategy_key(self.rows, 'ground_truth_position')
