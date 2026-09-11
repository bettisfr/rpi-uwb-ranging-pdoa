#!/usr/bin/env python3
"""Cross-flight diagnostic calibration of UAV horizontal ranges.

Models are trained on one manually flown trajectory and tested exclusively on
the other.  They are deliberately labelled ad-hoc: unlike the static model,
they use UAV ground truth and are not a deployment-time correction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis" / "outputs" / "uav_range_transfer" / "predictions.csv"
OUTPUT = ROOT / "analysis" / "outputs" / "uav_range_ad_hoc"


def summarize(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error)
    return {
        "samples": len(error), "bias_m": float(error.mean()), "mae_m": float(absolute.mean()),
        "median_absolute_error_m": float(np.median(absolute)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))), "p90_absolute_error_m": float(np.quantile(absolute, .9)),
    }


def fit_predict(name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if name == "raw":
        return test_x, {}
    if name == "mean_offset":
        offset = float(np.mean(train_y - train_x))
        return test_x + offset, {"offset_m": offset}
    if name == "median_offset":
        offset = float(np.median(train_y - train_x))
        return test_x + offset, {"offset_m": offset}
    degree = {"linear": 1, "quadratic": 2}[name]
    coefficients = np.polyfit(train_x, train_y, degree)
    return np.polyval(coefficients, test_x), {f"coefficient_{i}": float(value) for i, value in enumerate(coefficients)}


def main() -> None:
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    flights = sorted({row["flight"] for row in rows})
    if len(flights) != 2:
        raise ValueError(f"Expected two flights, found {flights}")
    methods = ("raw", "mean_offset", "median_offset", "linear", "quadratic")
    report = []
    for train_flight in flights:
        train = [row for row in rows if row["flight"] == train_flight]
        test = [row for row in rows if row["flight"] != train_flight]
        train_x = np.asarray([float(row["nearest_horizontal_truth_m"]) + float(row["nearest_raw_horizontal_error_m"]) for row in train])
        train_y = np.asarray([float(row["nearest_horizontal_truth_m"]) for row in train])
        test_x = np.asarray([float(row["nearest_horizontal_truth_m"]) + float(row["nearest_raw_horizontal_error_m"]) for row in test])
        test_y = np.asarray([float(row["nearest_horizontal_truth_m"]) for row in test])
        for method in methods:
            predicted, parameters = fit_predict(method, train_x, train_y, test_x)
            report.append({
                "train_flight": train_flight, "test_flight": test[0]["flight"], "method": method,
                **parameters, **summarize(predicted - test_y),
            })
    OUTPUT.mkdir(exist_ok=True)
    fields = sorted({field for row in report for field in row})
    with (OUTPUT / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader(); writer.writerows(report)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "complete", "protocol": "train one flight, test the other",
        "source": str(SOURCE), "purpose": "diagnostic UAV-only calibration",
    }, indent=2) + "\n")
    for row in report:
        print(row)


if __name__ == "__main__":
    main()
