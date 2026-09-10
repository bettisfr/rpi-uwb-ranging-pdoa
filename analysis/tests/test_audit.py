"""Small malformed/partial datasets exercise the audit before model fitting."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workflow.audit import audit_static
from workflow.data import STATIC_REQUIRED_COLUMNS


class AuditTests(unittest.TestCase):
    def row(self, **changes):
        row = dict.fromkeys(STATIC_REQUIRED_COLUMNS, "0")
        row.update(experiment_id="h0", run_file="same.csv", tag="dw00",
                   target_ground_distance_m="2", gt_bearing_deg="15",
                   range_m="2")
        row.update(changes)
        return row

    def audit(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(STATIC_REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            report = audit_static(path, Path(directory) / "output")
            with (Path(directory) / "output/static_configuration_inventory.csv").open() as stream:
                inventory = list(csv.DictReader(stream))
            return report, inventory

    def test_partial_coverage_and_identical_filenames_across_campaigns(self):
        report, inventory = self.audit([
            self.row(),
            self.row(experiment_id="h4", height_difference_m="4", range_m="3",
                     ground_distance_error_m=""),
        ])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["invalid_horizontal_projections"], 1)
        self.assertEqual(report["missing_configurations"], 1618)
        self.assertEqual(len(inventory), 1620)
        self.assertEqual(sum(row["coverage"] == "partial" for row in inventory), 2)

    def test_bad_numbers_geometry_and_mixed_run_are_reported(self):
        report, _ = self.audit([
            self.row(pdoa_deg="nan", ground_distance_error_m="1"),
            self.row(target_ground_distance_m="4", ground_distance_error_m="-2",
                     gt_bearing_deg="35"),
        ])
        self.assertEqual(report["status"], "failed")
        for code in ("invalid_numeric", "inconsistent_ground_target",
                     "mixed_run_factors", "tag_bearing_mismatch"):
            self.assertIn(code, report["error_counts"])

    def test_empty_dataset_is_not_a_success(self):
        report, _ = self.audit([])
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
