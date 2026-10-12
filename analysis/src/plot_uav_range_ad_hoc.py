#!/usr/bin/env python3
"""Plot cross-flight range residuals before and after UAV-only offsets."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis" / "outputs" / "uav_range_transfer" / "predictions.csv"
OUTPUT = ROOT / "analysis" / "plots" / "uav_range_ad_hoc_errors.pdf"


def main() -> None:
    with SOURCE.open(newline="") as source:
        rows = list(csv.DictReader(source))
    flights = sorted({row["flight"] for row in rows})
    offsets = {}
    for flight in flights:
        train = [row for row in rows if row["flight"] == flight]
        offsets[flight] = np.mean([
            float(row["horizontal_truth_m"]) - (
                float(row["horizontal_truth_m"]) + float(row["raw_horizontal_error_m"])
            )
            for row in train
        ])

    distance = np.asarray([float(row["horizontal_truth_m"]) for row in rows])
    raw = np.asarray([float(row["raw_horizontal_error_m"]) for row in rows])
    static = np.asarray([float(row["corrected_horizontal_error_m"]) for row in rows])
    # Each row is evaluated with the offset fitted only on the other flight.
    ad_hoc = np.asarray([raw[index] + offsets[next(f for f in flights if f != row["flight"])]
                         for index, row in enumerate(rows)])

    edges = np.arange(0., np.ceil(distance.max() / 2.) * 2. + 2., 2.)
    centers = (edges[:-1] + edges[1:]) / 2.
    figure, axis = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    for values, label, color, marker in (
        (raw, "Raw", "#707070", "o"),
        (static, "Static quadratic", "#2468a2", "s"),
        (ad_hoc, "UAV offset (cross-flight)", "#b02a82", "D"),
    ):
        means, deviations = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = values[(distance >= lower) & (distance < upper)]
            means.append(np.mean(selected) if len(selected) else np.nan)
            deviations.append(np.std(selected) if len(selected) else np.nan)
        axis.errorbar(centers, means, yerr=deviations, color=color, marker=marker,
                      markersize=3.5, linewidth=1.1, capsize=2, label=label)
    axis.axhline(0., color="black", linewidth=.7)
    axis.set_xlabel("RTK horizontal range (m)")
    axis.set_ylabel("Signed range error (m)")
    axis.grid(axis="y", linewidth=.4, alpha=.35)
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
