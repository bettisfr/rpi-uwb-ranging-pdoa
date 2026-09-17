#!/usr/bin/env python3
"""Diagnose UAV PDoA against RTK geometry with an oracle mount alignment.

This is explicitly a diagnostic, not a deployable PDoA calibration: the
constant node-to-aircraft yaw offset is selected using ground truth.  It tells
us whether a fixed mounting convention can explain the raw phase at all.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from uav.angles import wrap_degrees
from uav.config import ANALYSIS, UAV_PREDICTIONS, UAV_REFERENCE, tag_xy
from uav.coordinates import q4_heading_deg


PREDICTIONS = UAV_PREDICTIONS
REFERENCE = UAV_REFERENCE
OUTPUT = ANALYSIS / "outputs" / "uav_pdoa_diagnostic.json"


def phase_geometry(rows: list[dict[str, str]], q4_heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return target headings and horizontal/3D factors for all observations."""
    headings, factors = [], []
    for row in rows:
        tag_x, tag_y = tag_xy(row["tag"])
        dx, dy = tag_x - float(row["node_x_m"]), tag_y - float(row["node_y_m"])
        headings.append(q4_heading_deg + math.degrees(math.atan2(dx, dy)) - float(row["yaw_deg"]))
        factors.append(math.hypot(dx, dy) / float(row["true_range_m"]))
    return np.asarray(headings), np.asarray(factors)


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    q4_heading = q4_heading_deg(reference)
    with PREDICTIONS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    observed = np.asarray([float(row["pdoa_deg"]) for row in rows])
    relative_headings, factors = phase_geometry(rows, q4_heading)

    best = None
    for sign in (-1., 1.):
        for offset_deg in np.arange(-180., 180.01, .25):
            expected = sign * 180. * np.sin(np.radians(relative_headings - offset_deg)) * factors
            residual = wrap_degrees(observed - expected)
            score = float(np.mean(np.abs(residual)))
            if best is None or score < best[0]:
                best = score, float(offset_deg), sign, residual
    assert best is not None
    _, offset_deg, sign, residual = best
    absolute = np.abs(residual)
    report = {
        "status": "diagnostic_oracle_mount_alignment",
        "samples": len(rows),
        "q4_heading_from_north_deg": q4_heading,
        "best_node_to_aircraft_yaw_offset_deg": offset_deg,
        "antenna_order_sign": sign,
        "phase_residual_mae_deg": float(absolute.mean()),
        "phase_residual_median_absolute_deg": float(np.median(absolute)),
        "phase_residual_rmse_deg": float(np.sqrt(np.mean(residual ** 2))),
        "phase_residual_p90_absolute_deg": float(np.quantile(absolute, .9)),
        "phase_residual_max_absolute_deg": float(absolute.max()),
        "caveat": "The mount yaw offset is fitted with RTK ground truth and is diagnostic only.",
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
