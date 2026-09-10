"""Dataset-adapter boundary for normalized static and UAV measurement tables."""

from __future__ import annotations

import csv
from pathlib import Path


STATIC_REQUIRED_COLUMNS = {
    "experiment_id",
    "run_file",
    "tag",
    "height_difference_m",
    "target_ground_distance_m",
    "tag_rotation_deg",
    "gt_bearing_deg",
    "gt_x_m",
    "gt_y_m",
    "range_m",
    "pdoa_deg",
    "clk_ppm",
    "ground_distance_error_m",
    "eta_residual_deg",
    "position_error_m",
}


def validate_normalized_static_table(path: Path) -> None:
    """Fail early when the existing normalization no longer meets the contract."""
    with path.open(newline="") as source:
        columns = set(csv.DictReader(source).fieldnames or ())
    missing = STATIC_REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_static_rows(path: Path) -> list[dict[str, str]]:
    """Load the normalized static table after contract validation.

    Conversion to numerical arrays belongs to the future feature builder so the
    on-disk source remains inspectable and directly usable by CSV tools.
    """
    validate_normalized_static_table(path)
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def load_uav_rows(_: Path) -> list[dict[str, str]]:
    """Reserved adapter for the time-aligned UAV table.

    The UAV campaign must expose its own normalized schema before it can enter
    the benchmark workflow; it is deliberately not silently mixed with static
    measurements.
    """
    raise NotImplementedError("UAV normalization will be added before dynamic evaluation.")
