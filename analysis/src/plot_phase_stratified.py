#!/usr/bin/env python3
"""Visualize leave-one-distance-out PDoA calibration results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "pdoa_stratified_lodo"
CSV = ROOT / "analysis" / "csv"
PLOTS = ROOT / "analysis" / "plots"


METHOD_LABELS = {
    "raw": "Raw",
    "offset_global": "Global offset",
    "trigonometric_global": "Global trig.",
    "offset_bearing": "Bearing offset",
    "trigonometric_bearing": "Bearing trig.",
    "offset_rotation": "Rotation offset",
    "trigonometric_rotation": "Rotation trig.",
    "offset_bearing_rotation": "Bearing×rotation offset",
    "trigonometric_bearing_rotation": "Bearing×rotation trig.",
}


def wrapped_error(phase: pd.Series, expected: pd.Series) -> pd.Series:
    return (phase - expected + 180) % 360 - 180


def overall_plot(metrics: pd.DataFrame) -> None:
    metrics = metrics.set_index("method").loc[list(METHOD_LABELS)].reset_index()
    labels = [METHOD_LABELS[name] for name in metrics.method]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 4.5), sharey=True, layout="constrained")
    for axis, metric, title, color in zip(axes, ["mae_deg", "rmse_deg"], ["MAE", "RMSE"], ["#1565c0", "#8e24aa"]):
        axis.barh(y, metrics[metric], color=color, height=.66)
        axis.set_title(title)
        axis.set_xlabel("PDoA residual [deg]")
        axis.grid(axis="x", alpha=.25)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    figure.savefig(PLOTS / "pdoa_stratified_lodo_overall.pdf", bbox_inches="tight")
    plt.close(figure)


def tag_plot(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = ["raw", "trigonometric_global", "trigonometric_bearing", "trigonometric_bearing_rotation"]
    labels = [METHOD_LABELS[method] for method in methods]
    records = []
    for method in methods:
        error = wrapped_error(predictions[f"{method}_phase_deg"], predictions.expected_phase_deg)
        for tag, values in error.groupby(predictions.tag):
            records.append(dict(method=method, tag=tag, samples=len(values), mae_deg=values.abs().mean(),
                                rmse_deg=np.sqrt(np.mean(values**2))))
    summary = pd.DataFrame(records)
    tags = sorted(summary.tag.unique())
    x = np.arange(len(tags))
    width = .19
    colors = ["#616161", "#1565c0", "#00897b", "#8e24aa"]
    figure, axis = plt.subplots(figsize=(9.1, 4.1), layout="constrained")
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = summary[summary.method.eq(method)].set_index("tag").loc[tags, "mae_deg"]
        axis.bar(x + (index - 1.5) * width, values, width=width, color=color, label=label)
    axis.set_xticks(x, tags)
    axis.set_ylabel("PDoA residual MAE [deg]")
    axis.set_xlabel("Tag / deployment bearing")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.19))
    figure.savefig(PLOTS / "pdoa_stratified_lodo_by_tag.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    CSV.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUTPUT / "overall_metrics.csv")
    predictions = pd.read_csv(OUTPUT / "predictions.csv")
    overall_plot(metrics)
    summary = tag_plot(predictions)
    summary.to_csv(CSV / "pdoa_stratified_lodo_by_tag.csv", index=False)
    print(f"Wrote {PLOTS / 'pdoa_stratified_lodo_overall.pdf'}")
    print(f"Wrote {PLOTS / 'pdoa_stratified_lodo_by_tag.pdf'}")
    print(f"Wrote {CSV / 'pdoa_stratified_lodo_by_tag.csv'}")


if __name__ == "__main__":
    main()
