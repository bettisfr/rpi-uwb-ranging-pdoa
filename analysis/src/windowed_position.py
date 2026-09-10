#!/usr/bin/env python3
"""Evaluate position from temporally aggregated, out-of-fold static data.

Each prediction uses a non-overlapping consecutive window from one tag in one
static run.  The window itself observes only range and PDoA.  The range and
global-PDoA corrections are inherited from the existing leave-one-distance-out
predictions, so no held distance is used to fit its own correction.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from workflow.offset import load_assigned_rows
from workflow.position_benchmark import metrics, reconstruct, write_csv


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "analysis" / "csv" / "static_measurements.csv"
OUTPUT = ROOT / "analysis" / "outputs"


def circular_mean(values: list[float]) -> float:
    """Return a circular mean in degrees."""
    return math.degrees(math.atan2(
        sum(math.sin(math.radians(value)) for value in values),
        sum(math.cos(math.radians(value)) for value in values),
    ))


def circular_std(values: list[float]) -> float:
    """Return circular standard deviation in degrees."""
    sine = np.mean(np.sin(np.deg2rad(values)))
    cosine = np.mean(np.cos(np.deg2rad(values)))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(max(float(np.hypot(sine, cosine)), 1e-15)))))


def load_prediction_maps() -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    """Load the existing LODO predictions, keyed by normalized CSV line."""
    range_predictions: dict[int, dict[str, str]] = {}
    for path in sorted((OUTPUT / "distance_lodo").glob("distance_*m/predictions.csv")):
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                line = int(row["csv_line"])
                if line in range_predictions:
                    raise ValueError(f"Duplicate range prediction at CSV line {line}")
                range_predictions[line] = row
    with (OUTPUT / "pdoa_stratified_lodo" / "predictions.csv").open(newline="") as source:
        phase_predictions = {int(row["csv_line"]): row for row in csv.DictReader(source)}
    return range_predictions, phase_predictions


def summarize(name: str, source: list[dict], x_hat: np.ndarray, y_hat: np.ndarray,
              clipped: np.ndarray) -> list[dict]:
    """Report overall metrics plus raw-PDoA-stability gates."""
    x_true = np.asarray([row["x_true_m"] for row in source], dtype=float)
    y_true = np.asarray([row["y_true_m"] for row in source], dtype=float)
    stability = np.asarray([row["pdoa_std_deg"] for row in source], dtype=float)
    result = []
    for gate in (None, 10.0, 20.0):
        select = np.ones(len(source), dtype=bool) if gate is None else stability <= gate
        result.append(dict(
            method=name,
            pdoa_std_gate_deg="all" if gate is None else gate,
            accepted_windows=int(select.sum()),
            accepted_fraction=float(select.mean()),
            **metrics(x_true[select], y_true[select], x_hat[select], y_hat[select], clipped[select]),
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=20,
                        help="number of consecutive observations per aggregate")
    arguments = parser.parse_args()
    if arguments.window_size < 2:
        raise ValueError("window size must be at least two")
    rows, split, digest = load_assigned_rows(DATASET, OUTPUT)
    range_predictions, phase_predictions = load_prediction_maps()
    development = [row for row in rows if row["subset"] in ("train", "validation")]
    by_series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in development:
        by_series[(row["run_file"], row["tag"])].append(row)

    records: list[dict] = []
    window_size = arguments.window_size
    for (_, tag), observations in by_series.items():
        observations.sort(key=lambda row: int(row["sequence"]))
        for start in range(0, len(observations) - window_size + 1, window_size):
            window = observations[start : start + window_size]
            if any(int(row["csv_line"]) not in range_predictions or
                   int(row["csv_line"]) not in phase_predictions for row in window):
                continue
            raw_rho = [float(row["estimated_ground_distance_m"]) for row in window]
            if not all(math.isfinite(value) for value in raw_rho):
                continue
            corrected_rho = [float(range_predictions[int(row["csv_line"])]["quadratic_regression_m"])
                             for row in window]
            raw_phase = [float(row["pdoa_deg"]) for row in window]
            global_phase = [float(phase_predictions[int(row["csv_line"])]["trigonometric_global_phase_deg"])
                            for row in window]
            first = window[0]
            record = dict(
                run_file=first["run_file"], tag=tag, height_difference_m=float(first["height_difference_m"]),
                held_distance_m=float(first["target_ground_distance_m"]),
                x_true_m=float(first["gt_x_m"]), y_true_m=float(first["gt_y_m"]),
                pdoa_std_deg=circular_std(raw_phase),
                raw_rho_m=float(np.mean(raw_rho)), corrected_rho_m=float(np.mean(corrected_rho)),
                raw_phase_deg=circular_mean(raw_phase), global_phase_deg=circular_mean(global_phase),
                firmware_x_m=float(np.mean([float(row["x_m"]) for row in window])),
                firmware_y_m=float(np.mean([float(row["y_m"]) for row in window])),
            )
            records.append(record)

    if not records:
        raise ValueError("No complete consecutive windows with LODO predictions")
    destination = OUTPUT / f"windowed_position_lodo_w{window_size}"
    destination.mkdir(exist_ok=True)
    manifest = dict(
        status="running", protocol="leave_one_distance_out",
        source_sha256=digest, split=split, window_size=window_size,
        aggregation="arithmetic mean for horizontal range and firmware x/y; circular mean for PDoA",
        stability="circular standard deviation of raw PDoA within the consecutive window",
        allowed_inputs=["20 consecutive range observations", "20 consecutive PDoA observations", "height_difference_m"],
        forbidden_inputs=["tag", "bearing", "tag_rotation_deg", "ground-truth position"],
        correction_inputs="Existing per-observation LODO range-quadratic and global-PDoA predictions, then aggregated",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    height = np.asarray([row["height_difference_m"] for row in records])
    estimates = {
        "firmware_mean_xy": (np.asarray([row["firmware_x_m"] for row in records]),
                               np.asarray([row["firmware_y_m"] for row in records]),
                               np.zeros(len(records), dtype=bool)),
        f"geometry_raw_window{window_size}": reconstruct([row["raw_rho_m"] for row in records], height,
                                                           [row["raw_phase_deg"] for row in records]),
        f"geometry_range_quadratic_window{window_size}": reconstruct([row["corrected_rho_m"] for row in records], height,
                                                                       [row["raw_phase_deg"] for row in records]),
        f"geometry_phase_global_window{window_size}": reconstruct([row["raw_rho_m"] for row in records], height,
                                                                    [row["global_phase_deg"] for row in records]),
        f"geometry_joint_window{window_size}": reconstruct([row["corrected_rho_m"] for row in records], height,
                                                             [row["global_phase_deg"] for row in records]),
    }
    summary: list[dict] = []
    for name, (x_hat, y_hat, clipped) in estimates.items():
        summary.extend(summarize(name, records, x_hat, y_hat, clipped))
        for record, x_value, y_value, was_clipped in zip(records, x_hat, y_hat, clipped):
            record[f"{name}_x_m"] = float(x_value)
            record[f"{name}_y_m"] = float(y_value)
            record[f"{name}_clipped"] = bool(was_clipped)
    write_csv(destination / "window_predictions.csv", records)
    write_csv(destination / "overall_metrics.csv", summary)
    manifest["status"] = "complete"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in summary:
        if row["method"] == f"geometry_joint_window{window_size}":
            print(
                f"gate={row['pdoa_std_gate_deg']}: n={row['accepted_windows']} "
                f"({row['accepted_fraction']:.1%}), position MAE={row['position_mae_m']:.3f} m, "
                f"P90={row['position_p90_m']:.3f} m"
            )


if __name__ == "__main__":
    main()
