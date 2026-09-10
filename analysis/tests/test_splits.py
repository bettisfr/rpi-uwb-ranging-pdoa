"""Verify separation of correlated samples and repeatable split assignments."""

import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workflow.splits import build_grouped_split


class SplitTests(unittest.TestCase):
    def rows(self):
        return [dict(experiment_id=f"h{height}", run_file=f"d{distance}.csv",
                     tag=tag, height_difference_m=str(height),
                     target_ground_distance_m=str(distance), tag_rotation_deg="0",
                     samples="200")
                for height in (0, 2, 4) for distance in range(2, 32, 2)
                for tag in ("dw00", "dw04", "dw08")]

    def test_whole_conditions_and_repeated_runs_stay_together(self):
        rows = self.rows()
        rows.append({**rows[0], "run_file": "repeat.csv"})
        assigned = build_grouped_split(rows)
        subsets = defaultdict(set)
        for row in assigned:
            subsets[row["group_id"]].add(row["subset"])
        self.assertTrue(all(len(parts) == 1 for parts in subsets.values()))
        self.assertEqual(len(assigned), len(rows))
        for height in (0, 2, 4):
            observed = {row["subset"] for row in assigned if float(row["height_difference_m"]) == height}
            self.assertEqual(observed, {"train", "validation", "test"})

    def test_seed_and_input_order(self):
        rows = self.rows()
        self.assertEqual(build_grouped_split(rows), build_grouped_split(list(reversed(rows))))
        self.assertNotEqual(build_grouped_split(rows, seed=42), build_grouped_split(rows, seed=7))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            build_grouped_split(self.rows(), validation_fraction=0.6, test_fraction=0.5)
        with self.assertRaises(ValueError):
            build_grouped_split(self.rows()[:3])
        with self.assertRaises(ValueError):
            build_grouped_split([])
        rows = self.rows()
        rows.append({**rows[0], "tag_rotation_deg": "90"})
        with self.assertRaises(ValueError):
            build_grouped_split(rows)


if __name__ == "__main__":
    unittest.main()
