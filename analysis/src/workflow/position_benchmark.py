"""Out-of-fold generic position reconstruction from range and PDoA only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .offset import load_assigned_rows


def metrics(x_true, y_true, x_hat, y_hat, clipped):
    """Report Cartesian and horizontal-position errors without circular metrics."""
    ex, ey = x_hat - x_true, y_hat - y_true
    error = np.hypot(ex, ey)
    return dict(samples=len(error), x_bias_m=float(ex.mean()), y_bias_m=float(ey.mean()),
                x_mae_m=float(np.abs(ex).mean()), y_mae_m=float(np.abs(ey).mean()),
                position_mae_m=float(error.mean()), position_rmse_m=float(np.sqrt(np.mean(error**2))),
                position_p90_m=float(np.quantile(error, .9, method="inverted_cdf")),
                position_max_m=float(error.max()), physical_projection_rate=float(1 - clipped.mean()),
                clipped_samples=int(clipped.sum()))


def reconstruct(ground_distance, height, phase_deg):
    """Recover x/y from corrected horizontal range and PDoA phase.

    The test-bed occupies the positive-y half-plane.  A noisy phase can imply
    an impossible |x| > rho; projecting it onto the physical disk is explicit
    and reported, rather than silently producing NaNs.
    """
    rho = np.asarray(ground_distance, dtype=float)
    r3d = np.hypot(rho, np.asarray(height, dtype=float))
    unconstrained_x = -np.asarray(phase_deg, dtype=float) * r3d / 180.
    x_hat = np.clip(unconstrained_x, -rho, rho)
    clipped = np.abs(unconstrained_x) > rho
    y_hat = np.sqrt(np.maximum(0., rho**2 - x_hat**2))
    return x_hat, y_hat, clipped


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_generic_position_benchmark(dataset: Path, output: Path) -> list[dict]:
    """Compare firmware and generic geometry-only position estimates.

    Inputs are measured range, PDoA, and known height difference.  Corrected
    values come only from their matching leave-one-distance-out predictions.
    No tag ID, bearing, rotation, or ground-truth coordinate selects a model.
    """
    rows, split, digest = load_assigned_rows(dataset, output)
    rows_by_line = {int(row["csv_line"]): row for row in rows}
    range_directory = output / "distance_lodo"
    phase_path = output / "pdoa_stratified_lodo" / "predictions.csv"
    range_paths = sorted(range_directory.glob("distance_*m/predictions.csv"))
    if not range_paths or not phase_path.exists():
        raise FileNotFoundError("Run distance LODO and stratified PDoA LODO before position evaluation")
    range_rows = {}
    for range_path in range_paths:
        with range_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                line = int(row["csv_line"])
                if line in range_rows:
                    raise ValueError(f"Duplicate LODO range prediction for CSV line {line}")
                range_rows[line] = row
    with phase_path.open(newline="") as stream:
        phase_rows = {int(row["csv_line"]): row for row in csv.DictReader(stream)}
    common = sorted(set(range_rows) & set(phase_rows) & set(rows_by_line))
    if not common:
        raise ValueError("No rows shared by out-of-fold range and PDoA predictions")
    destination = output / "position_generic_lodo"
    destination.mkdir(exist_ok=True)
    manifest = dict(status="running", protocol="leave_one_distance_out",
                    source_sha256=digest, split=split, samples=len(common),
                    range_prediction="quadratic_regression_m from distance_lodo",
                    phase_prediction="trigonometric_global_phase_deg from pdoa_stratified_lodo",
                    allowed_inputs=["range_m", "pdoa_deg", "height_difference_m"],
                    forbidden_inputs=["tag", "bearing", "tag_rotation_deg", "ground-truth position"],
                    half_plane="positive y, as defined by the static test-bed",
                    infeasible_phase_policy="Project implied x onto [-rho, rho] and report clipping rate")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    values = {"x_true": [], "y_true": [], "height": [], "tag": [], "distance": [],
              "firmware_x": [], "firmware_y": [], "raw_x": [], "raw_y": [], "raw_clipped": [],
              "range_x": [], "range_y": [], "range_clipped": [], "phase_x": [], "phase_y": [],
              "phase_clipped": [], "joint_x": [], "joint_y": [], "joint_clipped": []}
    predictions = []
    for line in common:
        source, rang, phase = rows_by_line[line], range_rows[line], phase_rows[line]
        height = float(source["height_difference_m"])
        raw_rho = float(rang["raw_m"])
        corrected_rho = float(rang["quadratic_regression_m"])
        raw_phase = float(phase["raw_phase_deg"])
        global_phase = float(phase["trigonometric_global_phase_deg"])
        estimates = {
            "raw": reconstruct(raw_rho, height, raw_phase),
            "range": reconstruct(corrected_rho, height, raw_phase),
            "phase": reconstruct(raw_rho, height, global_phase),
            "joint": reconstruct(corrected_rho, height, global_phase),
        }
        x_true, y_true = float(source["gt_x_m"]), float(source["gt_y_m"])
        values["x_true"].append(x_true); values["y_true"].append(y_true)
        values["height"].append(height); values["tag"].append(source["tag"])
        values["distance"].append(float(source["target_ground_distance_m"]))
        values["firmware_x"].append(float(source["x_m"])); values["firmware_y"].append(float(source["y_m"]))
        record = dict(csv_line=line, tag=source["tag"], height_difference_m=height,
                      held_distance_m=float(rang["target_ground_distance_m"]),
                      x_true_m=x_true, y_true_m=y_true,
                      firmware_x_m=float(source["x_m"]), firmware_y_m=float(source["y_m"]))
        for name, (x_hat, y_hat, clipped) in estimates.items():
            values[f"{name}_x"].append(float(x_hat)); values[f"{name}_y"].append(float(y_hat)); values[f"{name}_clipped"].append(bool(clipped))
            record.update({f"{name}_x_m": float(x_hat), f"{name}_y_m": float(y_hat), f"{name}_phase_clipped": bool(clipped)})
        predictions.append(record)
    arrays = {name: np.asarray(value) for name, value in values.items()}
    models = {
        "firmware_raw_xy": (arrays["firmware_x"], arrays["firmware_y"], np.zeros(len(common), dtype=bool)),
        "geometry_raw": (arrays["raw_x"], arrays["raw_y"], arrays["raw_clipped"]),
        "geometry_range_quadratic": (arrays["range_x"], arrays["range_y"], arrays["range_clipped"]),
        "geometry_phase_global": (arrays["phase_x"], arrays["phase_y"], arrays["phase_clipped"]),
        "geometry_range_quadratic_phase_global": (arrays["joint_x"], arrays["joint_y"], arrays["joint_clipped"]),
    }
    summary, by_height = [], []
    for name, (x_hat, y_hat, clipped) in models.items():
        summary.append(dict(method=name, **metrics(arrays["x_true"], arrays["y_true"], x_hat, y_hat, clipped)))
        for height in sorted(set(arrays["height"])):
            selected = arrays["height"] == height
            by_height.append(dict(method=name, height_difference_m=height,
                                  **metrics(arrays["x_true"][selected], arrays["y_true"][selected],
                                            x_hat[selected], y_hat[selected], clipped[selected])))
    write_csv(destination / "predictions.csv", predictions)
    write_csv(destination / "overall_metrics.csv", summary)
    write_csv(destination / "metrics_by_height.csv", by_height)
    manifest["status"] = "complete"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary
