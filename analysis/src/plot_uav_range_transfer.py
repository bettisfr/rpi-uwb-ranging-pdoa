#!/usr/bin/env python3
"""Plot diagnostic binned range-transfer residuals by flight and geometry."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis" / "outputs" / "uav_range_transfer" / "predictions.csv"
DESTINATION = ROOT / "analysis" / "plots" / "uav_range_transfer_by_geometry.pdf"
COLORS = ("#1464b4", "#c23a89")


def binned_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, means, and standard deviations for populated bins."""
    centers, means, spread = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (x >= lower) & (x < upper)
        if not selected.any():
            continue
        centers.append((lower + upper) / 2)
        means.append(y[selected].mean())
        spread.append(y[selected].std())
    return np.asarray(centers), np.asarray(means), np.asarray(spread)


def main() -> None:
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True, layout="constrained")
    for color, flight in zip(COLORS, sorted({row["flight"] for row in rows})):
        selected = [row for row in rows if row["flight"] == flight]
        distance = np.asarray([float(row["horizontal_truth_m"]) for row in selected])
        height = np.asarray([float(row["node_height_m"]) for row in selected])
        error = np.asarray([float(row["corrected_horizontal_error_m"]) for row in selected])
        label = "Flight 1" if flight.endswith("131334") else "Flight 2"
        for axis, values, edges, xlabel in (
            (axes[0], distance, np.arange(0, 41, 4), "True horizontal distance [m]"),
            (axes[1], height, np.arange(0, 9, 1), "Node height [m]"),
        ):
            center, mean, deviation = binned_mean(values, error, edges)
            axis.errorbar(center, mean, yerr=deviation, color=color, marker="o", markersize=3,
                          linewidth=1.1, capsize=2, label=label)
            axis.set_xlabel(xlabel)
            axis.grid(alpha=.25, linewidth=.5)
    for axis in axes:
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_ylim(-3.5, 1.5)
    axes[0].set_ylabel("Static-corrected horizontal-range error [m]")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    DESTINATION.parent.mkdir(exist_ok=True)
    figure.savefig(DESTINATION, bbox_inches="tight")
    print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
