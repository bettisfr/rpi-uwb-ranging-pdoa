#!/usr/bin/env python3
"""Diagnose the frozen static range-transfer residual against UAV dynamics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis" / "outputs" / "uav_range_transfer" / "predictions.csv"
CSV_DESTINATION = ROOT / "analysis" / "csv" / "uav_range_dynamics.csv"
PLOT_DESTINATION = ROOT / "analysis" / "plots" / "uav_range_dynamics.pdf"
COLORS = ("#1464b4", "#c23a89")


def summarize_bins(values: np.ndarray, error: np.ndarray, edges: np.ndarray) -> list[dict[str, float]]:
    """Summarize populated half-open bins of one explanatory variable."""
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (values >= lower) & (values < upper)
        if selected.any():
            rows.append(dict(lower=lower, upper=upper, samples=int(selected.sum()),
                             error_bias_m=float(error[selected].mean()),
                             error_mae_m=float(np.abs(error[selected]).mean()),
                             error_std_m=float(error[selected].std())))
    return rows


def main() -> None:
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    records = [row for row in rows if np.isfinite(float(row["speed_mps"]))]
    error = np.asarray([float(row["corrected_horizontal_error_m"]) for row in records])
    speed = np.asarray([float(row["speed_mps"]) for row in records])
    height = np.asarray([float(row["node_height_m"]) for row in records])
    yaw = np.asarray([float(row["yaw_deg"]) % 360. for row in records])
    correlations = {
        "speed": float(np.corrcoef(speed, error)[0, 1]),
        "height": float(np.corrcoef(height, error)[0, 1]),
        "sin_yaw": float(np.corrcoef(np.sin(np.deg2rad(yaw)), error)[0, 1]),
        "cos_yaw": float(np.corrcoef(np.cos(np.deg2rad(yaw)), error)[0, 1]),
    }
    summaries = []
    for variable, values, edges in (
        ("speed_mps", speed, np.arange(0, 14, 1)),
        ("height_m", height, np.arange(0, 9, 1)),
        ("yaw_deg", yaw, np.arange(0, 361, 30)),
    ):
        for row in summarize_bins(values, error, edges):
            summaries.append(dict(variable=variable, **row))
    CSV_DESTINATION.parent.mkdir(exist_ok=True)
    with CSV_DESTINATION.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.65), sharey=True, layout="constrained")
    definitions = (
        ("Speed [m/s]", speed, np.arange(0, 14, 1)),
        ("Node height [m]", height, np.arange(0, 9, 1)),
        ("Yaw [deg]", yaw, np.arange(0, 361, 30)),
    )
    for axis, (label, values, edges) in zip(axes, definitions):
        for color, flight in zip(COLORS, sorted({row["flight"] for row in records})):
            selected = np.asarray([row["flight"] == flight for row in records])
            bins = summarize_bins(values[selected], error[selected], edges)
            centers = [(row["lower"] + row["upper"]) / 2 for row in bins]
            means = [row["error_bias_m"] for row in bins]
            stds = [row["error_std_m"] for row in bins]
            flight_label = "Flight 1" if flight.endswith("131334") else "Flight 2"
            axis.errorbar(centers, means, yerr=stds, color=color, marker="o", markersize=3,
                          linewidth=1.1, capsize=2, label=flight_label)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_xlabel(label)
        axis.grid(alpha=.25, linewidth=.5)
    axes[0].set_ylabel("Static-corrected range error [m]")
    axes[2].legend(frameon=False, fontsize=7, loc="lower left")
    PLOT_DESTINATION.parent.mkdir(exist_ok=True)
    figure.savefig(PLOT_DESTINATION, bbox_inches="tight")
    print(f"Correlation coefficients: {correlations}")
    print(f"Wrote {CSV_DESTINATION}")
    print(f"Wrote {PLOT_DESTINATION}")


if __name__ == "__main__":
    main()
