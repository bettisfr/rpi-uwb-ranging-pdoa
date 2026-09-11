#!/usr/bin/env python3
"""Evaluate firmware bearing and corrected-range position on UAV records.

The firmware bearing is derived from its instantaneous x/y output.  A single
node-to-aircraft yaw offset is calibrated on one flight and transferred to the
other; tag identity and surveyed tag coordinates are never inference inputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from recalculate_uav_local_frame import enu


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "analysis" / "outputs" / "uav_range_transfer" / "predictions.csv"
REFERENCE = ROOT / "data" / "uav-20260903" / "reference.json"
OUTPUT = ROOT / "analysis" / "outputs" / "uav_bearing_position"


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.) % 360. - 180.


def circular_mean_degrees(values: np.ndarray) -> float:
    return float(math.degrees(math.atan2(np.sin(np.radians(values)).mean(), np.cos(np.radians(values)).mean())))


def metrics(
    angle_error: np.ndarray,
    position_error: np.ndarray,
    oracle_range_position_error: np.ndarray,
) -> dict[str, float]:
    angle_abs = np.abs(angle_error)
    return {
        "samples": len(angle_error),
        "bearing_mae_deg": float(angle_abs.mean()),
        "bearing_median_absolute_deg": float(np.median(angle_abs)),
        "bearing_p90_absolute_deg": float(np.quantile(angle_abs, .9)),
        "position_mae_m": float(position_error.mean()),
        "position_median_m": float(np.median(position_error)),
        "position_p90_m": float(np.quantile(position_error, .9)),
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_oracle_range_mae_m": float(oracle_range_position_error.mean()),
        "position_oracle_range_median_m": float(np.median(oracle_range_position_error)),
        "position_oracle_range_p90_m": float(np.quantile(oracle_range_position_error, .9)),
    }


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    east, north = enu(reference["origin_lat_deg"], reference["origin_lon_deg"],
                      reference["q4_lat_deg"], reference["q4_lon_deg"])
    q4_heading_deg = math.degrees(math.atan2(east, north))
    with PREDICTIONS.open(newline="") as source:
        source_rows = list(csv.DictReader(source))

    rows = [row for row in source_rows if math.hypot(float(row["firmware_x_m"]), float(row["firmware_y_m"])) > 1e-6]
    firmware_theta = np.asarray([
        math.degrees(math.atan2(float(row["firmware_y_m"]), float(row["firmware_x_m"])))
        for row in rows
    ])
    yaw_relative = np.asarray([float(row["yaw_deg"]) - q4_heading_deg for row in rows])
    ground_theta = np.asarray([
        math.degrees(math.atan2(
            float(row["tag_y_m"]) - float(row["nearest_node_y_m"]),
            float(row["tag_x_m"]) - float(row["nearest_node_x_m"]),
        ))
        for row in rows
    ])
    uncalibrated = firmware_theta - yaw_relative
    flight = np.asarray([row["flight"] for row in rows])

    def evaluate(train_mask: np.ndarray, label: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        offset = circular_mean_degrees(ground_theta[train_mask] - uncalibrated[train_mask])
        test_mask = ~train_mask if label != "pooled_oracle" else train_mask
        estimated_theta = uncalibrated[test_mask] + offset
        angle_error = wrap_degrees(estimated_theta - ground_theta[test_mask])
        radius = np.asarray([float(rows[index]["nearest_corrected_horizontal_m"])
                             for index in np.flatnonzero(test_mask)])
        node_x = np.asarray([float(rows[index]["nearest_node_x_m"]) for index in np.flatnonzero(test_mask)])
        node_y = np.asarray([float(rows[index]["nearest_node_y_m"]) for index in np.flatnonzero(test_mask)])
        tag_x = np.asarray([float(rows[index]["tag_x_m"]) for index in np.flatnonzero(test_mask)])
        tag_y = np.asarray([float(rows[index]["tag_y_m"]) for index in np.flatnonzero(test_mask)])
        estimate_x = node_x + radius * np.cos(np.radians(estimated_theta))
        estimate_y = node_y + radius * np.sin(np.radians(estimated_theta))
        position_error = np.hypot(estimate_x - tag_x, estimate_y - tag_y)
        oracle_radius = np.asarray([float(rows[index]["nearest_horizontal_truth_m"])
                                    for index in np.flatnonzero(test_mask)])
        oracle_x = node_x + oracle_radius * np.cos(np.radians(estimated_theta))
        oracle_y = node_y + oracle_radius * np.sin(np.radians(estimated_theta))
        oracle_range_position_error = np.hypot(oracle_x - tag_x, oracle_y - tag_y)
        report = {
            "protocol": label,
            "yaw_offset_deg": offset,
            **metrics(angle_error, position_error, oracle_range_position_error),
        }
        records = []
        for local_index, source_index in enumerate(np.flatnonzero(test_mask)):
            records.append({
                "protocol": label, "flight": rows[source_index]["flight"], "tag": rows[source_index]["tag"],
                "bearing_ground_truth_deg": ground_theta[source_index],
                "bearing_estimate_deg": estimated_theta[local_index], "bearing_error_deg": angle_error[local_index],
                "range_corrected_m": radius[local_index], "position_error_m": position_error[local_index],
            })
        return report, records

    flights = sorted(set(flight))
    reports, records = [], []
    report, values = evaluate(np.ones(len(rows), dtype=bool), "pooled_oracle")
    reports.append(report); records.extend(values)
    for train_flight in flights:
        report, values = evaluate(flight == train_flight, f"train_{train_flight}_test_other_flight")
        reports.append(report); records.extend(values)

    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(reports[0]))
        writer.writeheader(); writer.writerows(reports)
    with (OUTPUT / "predictions.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "complete", "samples": len(rows), "excluded_zero_firmware_vector": len(source_rows) - len(rows),
        "q4_heading_from_north_deg": q4_heading_deg,
        "bearing": "atan2(firmware_y_m, firmware_x_m), then one constant yaw offset",
        "range": "static quadratic correction transferred without UAV fitting",
    }, indent=2) + "\n")
    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
