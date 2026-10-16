#!/usr/bin/env python3
"""Cross-flight diagnostic of UAV range residuals versus viewing geometry.

Relative bearing is reconstructed from RTK/tag ground truth.  Consequently
these models test physical explanatory power only; they are not deployable
calibrators for an unknown tag position.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from sklearn.linear_model import HuberRegressor

from uav.config import ANALYSIS, UAV_PREDICTIONS, UAV_REFERENCE
from uav.coordinates import q4_heading_deg


SOURCE = UAV_PREDICTIONS
REFERENCE = UAV_REFERENCE
OUTPUT = ANALYSIS / "outputs" / "uav_geometry_residual"


def feature_matrix(rows: list[dict[str, str]], q4_heading_deg: float, model: str) -> np.ndarray:
    relative, yaw, distance = [], [], []
    for row in rows:
        dx = float(row["tag_x_m"]) - float(row["node_x_m"])
        dy = float(row["tag_y_m"]) - float(row["node_y_m"])
        target_heading = q4_heading_deg + math.degrees(math.atan2(dx, dy))
        relative.append(math.radians(target_heading - float(row["yaw_deg"])))
        yaw.append(math.radians(float(row["yaw_deg"])))
        distance.append(float(row["horizontal_truth_m"]))
    relative, yaw, distance = np.asarray(relative), np.asarray(yaw), np.asarray(distance)
    columns = [np.ones(len(rows))]
    if model in ("relative", "relative_yaw", "relative_yaw_distance"):
        columns.extend((np.sin(relative), np.cos(relative), np.sin(2 * relative), np.cos(2 * relative)))
    if model in ("relative_yaw", "relative_yaw_distance"):
        columns.extend((np.sin(yaw), np.cos(yaw)))
    if model == "relative_yaw_distance":
        columns.extend((distance, distance ** 2))
    return np.column_stack(columns)


def summarize(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error)
    return {
        "samples": len(error), "bias_m": float(error.mean()), "mae_m": float(absolute.mean()),
        "median_absolute_error_m": float(np.median(absolute)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))), "p90_absolute_error_m": float(np.quantile(absolute, .9)),
    }


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    q4_heading = q4_heading_deg(reference)
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    flights = sorted({row["flight"] for row in rows})
    target = np.asarray([float(row["raw_horizontal_error_m"]) for row in rows])
    flight = np.asarray([row["flight"] for row in rows])
    report = []
    for train_flight in flights:
        train_mask = flight == train_flight
        test_mask = ~train_mask
        for name in ("offset", "relative", "relative_yaw", "relative_yaw_distance"):
            if name == "offset":
                correction = np.full(test_mask.sum(), -target[train_mask].mean())
            else:
                train_x = feature_matrix([row for row, keep in zip(rows, train_mask) if keep], q4_heading, name)
                test_x = feature_matrix([row for row, keep in zip(rows, test_mask) if keep], q4_heading, name)
                model = HuberRegressor(fit_intercept=False, alpha=0., max_iter=1000).fit(train_x, target[train_mask])
                correction = -model.predict(test_x)
            error = target[test_mask] + correction
            report.append({"train_flight": train_flight, "test_flight": flights[0] if train_flight == flights[1] else flights[1],
                           "model": name, **summarize(error)})
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "complete", "protocol": "train one flight, test the other",
        "caveat": "Relative bearing is ground-truth-derived and diagnostic only.",
        "q4_heading_from_north_deg": q4_heading,
    }, indent=2) + "\n")
    for row in report:
        print(row)


if __name__ == "__main__":
    main()
