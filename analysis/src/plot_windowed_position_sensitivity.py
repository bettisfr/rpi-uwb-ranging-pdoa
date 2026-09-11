#!/usr/bin/env python3
"""Plot the grouped validation/test coverage--accuracy curves for PDoA gates."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis" / "outputs" / "windowed_position_sensitivity" / "coverage_accuracy.csv"
DESTINATION = ROOT / "analysis" / "plots" / "windowed_position_coverage_accuracy.pdf"
COLORS = {10: "#1f77b4", 20: "#e377c2", 30: "#2ca02c"}


def main() -> None:
    with SOURCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True, layout="constrained")
    for axis, subset in zip(axes, ("validation", "test")):
        for window_size in (10, 20, 30):
            selected = [row for row in rows if row["subset"] == subset and
                        int(row["window_size"]) == window_size and row["pdoa_std_gate_deg"] != "all"]
            selected.sort(key=lambda row: float(row["pdoa_std_gate_deg"]))
            coverage = [100 * float(row["accepted_fraction"]) for row in selected]
            mae = [float(row["position_mae_m"]) for row in selected]
            axis.plot(coverage, mae, marker="o", markersize=3.2, linewidth=1.1,
                      color=COLORS[window_size], label=f"$K={window_size}$")
            for row, x_value, y_value in zip(selected, coverage, mae):
                if float(row["pdoa_std_gate_deg"]) in (10., 20.):
                    axis.annotate(f"{float(row['pdoa_std_gate_deg']):g}°", (x_value, y_value),
                                  xytext=(3, 3), textcoords="offset points", fontsize=6)
        axis.set_title(subset.capitalize(), fontsize=9)
        axis.set_xlabel("Accepted windows [%]")
        axis.grid(alpha=.25, linewidth=.5)
        axis.set_xlim(15, 100)
    axes[0].set_ylabel("Position MAE [m]")
    axes[1].legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.4)
    DESTINATION.parent.mkdir(exist_ok=True)
    figure.savefig(DESTINATION, bbox_inches="tight")
    print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
