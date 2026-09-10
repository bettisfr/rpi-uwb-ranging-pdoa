#!/usr/bin/env python3
"""Visualize generic out-of-fold position-reconstruction results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "position_generic_lodo"
CSV = ROOT / "analysis" / "csv"
PLOTS = ROOT / "analysis" / "plots"

METHODS = ["firmware_raw_xy", "geometry_raw", "geometry_range_quadratic",
           "geometry_phase_global", "geometry_range_quadratic_phase_global"]
LABELS = {"firmware_raw_xy": "Firmware", "geometry_raw": "Geometry raw",
          "geometry_range_quadratic": "Range corr.", "geometry_phase_global": "PDoA global",
          "geometry_range_quadratic_phase_global": "Range + PDoA"}
COLORS = {"firmware_raw_xy": "#616161", "geometry_raw": "#1565c0",
          "geometry_range_quadratic": "#00897b", "geometry_phase_global": "#ef6c00",
          "geometry_range_quadratic_phase_global": "#8e24aa"}


def overall_plot(summary: pd.DataFrame) -> None:
    selected = summary.set_index("method").loc[METHODS]
    y = np.arange(len(METHODS))
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True, layout="constrained")
    for axis, column, title in zip(axes, ["position_mae_m", "position_p90_m"], ["Position MAE", "Position P90"]):
        axis.barh(y, selected[column], color=[COLORS[name] for name in METHODS], height=.62)
        axis.set_title(title)
        axis.set_xlabel("Position error [m]")
        axis.grid(axis="x", alpha=.25)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, [LABELS[name] for name in METHODS])
    axes[0].invert_yaxis()
    figure.savefig(PLOTS / "position_generic_lodo_overall.pdf", bbox_inches="tight")
    plt.close(figure)


def tag_plot(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method in METHODS:
        if method == "firmware_raw_xy":
            x_hat, y_hat = predictions.firmware_x_m, predictions.firmware_y_m
        else:
            prefix = {"geometry_raw": "raw", "geometry_range_quadratic": "range",
                      "geometry_phase_global": "phase", "geometry_range_quadratic_phase_global": "joint"}[method]
            x_hat, y_hat = predictions[f"{prefix}_x_m"], predictions[f"{prefix}_y_m"]
        error = np.hypot(x_hat - predictions.x_true_m, y_hat - predictions.y_true_m)
        for tag, values in pd.Series(error.to_numpy(), index=predictions.tag).groupby(level=0):
            records.append(dict(method=method, tag=tag, samples=len(values), position_mae_m=values.mean(),
                                position_rmse_m=np.sqrt(np.mean(values**2))))
    summary = pd.DataFrame(records)
    visible = ["firmware_raw_xy", "geometry_raw", "geometry_range_quadratic_phase_global"]
    tags = sorted(summary.tag.unique())
    x, width = np.arange(len(tags)), .25
    figure, axis = plt.subplots(figsize=(8.6, 3.9), layout="constrained")
    for index, method in enumerate(visible):
        values = summary[summary.method.eq(method)].set_index("tag").loc[tags, "position_mae_m"]
        axis.bar(x + (index - 1) * width, values, width=width, color=COLORS[method], label=LABELS[method])
    axis.set_xticks(x, tags)
    axis.set_xlabel("Tag / deployment bearing")
    axis.set_ylabel("Position MAE [m]")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18))
    figure.savefig(PLOTS / "position_generic_lodo_by_tag.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    CSV.mkdir(exist_ok=True)
    overall_plot(pd.read_csv(OUTPUT / "overall_metrics.csv"))
    by_tag = tag_plot(pd.read_csv(OUTPUT / "predictions.csv"))
    by_tag.to_csv(CSV / "position_generic_lodo_by_tag.csv", index=False)
    print(f"Wrote {PLOTS / 'position_generic_lodo_overall.pdf'}")
    print(f"Wrote {PLOTS / 'position_generic_lodo_by_tag.pdf'}")
    print(f"Wrote {CSV / 'position_generic_lodo_by_tag.csv'}")


if __name__ == "__main__":
    main()
