#!/usr/bin/env python3
"""Show raw firmware x/y coordinates directly against their ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "analysis" / "csv" / "static_measurements.csv"
CSV = ROOT / "analysis" / "csv"
PLOTS = ROOT / "analysis" / "plots"


def main() -> None:
    CSV.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    rows = pd.read_csv(DATASET)
    groups = ["experiment_id", "run_file", "tag", "height_difference_m",
              "target_ground_distance_m", "tag_rotation_deg", "gt_x_m", "gt_y_m"]
    means = rows.groupby(groups, as_index=False).agg(x_hat_m=("x_m", "mean"),
                                                       y_hat_m=("y_m", "mean"),
                                                       samples=("x_m", "size"))
    means.to_csv(CSV / "raw_firmware_position_means.csv", index=False)
    tags = sorted(means.tag.unique())
    palette = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), layout="constrained")
    for axis, coordinate, estimate in zip(axes, ["x", "y"], ["x_hat_m", "y_hat_m"]):
        truth = f"gt_{coordinate}_m"
        for index, tag in enumerate(tags):
            selected = means[means.tag.eq(tag)]
            axis.scatter(selected[truth], selected[estimate], s=9, alpha=.48,
                         color=palette(index), label=tag)
        minimum = min(means[truth].min(), means[estimate].min())
        maximum = max(means[truth].max(), means[estimate].max())
        margin = .04 * (maximum - minimum)
        identity = np.array([minimum-margin, maximum+margin])
        axis.plot(identity, identity, "--", color="black", linewidth=1, label="ideal")
        slope, intercept = np.polyfit(means[truth], means[estimate], 1)
        axis.plot(identity, slope * identity + intercept, color="#e53935", linewidth=1.2)
        axis.text(.04, .94, fr"fit: $\hat{{{coordinate}}}={slope:.2f}{coordinate}{intercept:+.2f}$",
                  transform=axis.transAxes, va="top", color="#c62828")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(fr"${{{coordinate}}}_{{\mathrm{{GT}}}}$ [m]")
        axis.set_ylabel(fr"Firmware $\hat{{{coordinate}}}$ [m]")
        axis.set_title(fr"Raw firmware ${coordinate}$ coordinate")
        axis.grid(alpha=.25)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.12))
    figure.savefig(PLOTS / "raw_firmware_coordinates.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {CSV / 'raw_firmware_position_means.csv'}")
    print(f"Wrote {PLOTS / 'raw_firmware_coordinates.pdf'}")


if __name__ == "__main__":
    main()
