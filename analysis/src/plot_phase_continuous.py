#!/usr/bin/env python3
"""Visualize continuous generic PDoA results on held-out bearings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "pdoa_continuous_lobo"
PLOTS = ROOT / "analysis" / "plots"
METHODS = ["raw", "trigonometric_global", "harmonic_range_height", "mlp_range_height_cuda"]
LABELS = {"raw": "Raw", "trigonometric_global": "Global trig.",
          "harmonic_range_height": "Harmonic + range/h", "mlp_range_height_cuda": "MLP + range/h"}
COLORS = {"raw": "#616161", "trigonometric_global": "#1565c0",
          "harmonic_range_height": "#ef6c00", "mlp_range_height_cuda": "#e53935"}


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    overall = pd.read_csv(OUTPUT / "overall_metrics.csv").set_index("method").loc[METHODS]
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharey=True, layout="constrained")
    y = np.arange(len(METHODS))
    for axis, metric, title in zip(axes, ["mae_deg", "rmse_deg"], ["MAE", "RMSE"]):
        axis.barh(y, overall[metric], color=[COLORS[m] for m in METHODS], height=.62)
        axis.set_title(title)
        axis.set_xlabel("PDoA residual [deg]")
        axis.grid(axis="x", alpha=.25)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, [LABELS[m] for m in METHODS]); axes[0].invert_yaxis()
    figure.savefig(PLOTS / "pdoa_continuous_lobo_overall.pdf", bbox_inches="tight")
    plt.close(figure)
    metrics = pd.read_csv(OUTPUT / "metrics.csv")
    tags = metrics.sort_values("held_bearing_deg").held_tag.unique()
    x, width = np.arange(len(tags)), .19
    figure, axis = plt.subplots(figsize=(8.8, 3.9), layout="constrained")
    for i, method in enumerate(METHODS):
        values = metrics[metrics.method.eq(method)].set_index("held_tag").loc[tags, "mae_deg"]
        axis.bar(x + (i - 1.5)*width, values, width=width, color=COLORS[method], label=LABELS[method])
    axis.set_xticks(x, tags)
    axis.set_xlabel("Held-out tag / bearing")
    axis.set_ylabel("PDoA residual MAE [deg]")
    axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18))
    figure.savefig(PLOTS / "pdoa_continuous_lobo_by_bearing.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {PLOTS / 'pdoa_continuous_lobo_overall.pdf'}")
    print(f"Wrote {PLOTS / 'pdoa_continuous_lobo_by_bearing.pdf'}")


if __name__ == "__main__":
    main()
