#!/usr/bin/env python3
"""Export raw PDoA residual summaries and tag/rotation diagnostic plots."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "analysis" / "csv" / "static_measurements.csv"
CSV_DIRECTORY = ROOT / "analysis" / "csv"
PLOTS_DIRECTORY = ROOT / "analysis" / "plots"


def circular_bias(values: pd.Series) -> float:
    radians = np.deg2rad(values.to_numpy(float))
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * radians)))))


def circular_std(values: pd.Series) -> float:
    radians = np.deg2rad(values.to_numpy(float))
    resultant = abs(np.mean(np.exp(1j * radians)))
    return float(np.rad2deg(np.sqrt(-2 * np.log(max(resultant, 1e-15)))))


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    for key, group in frame.groupby(groups, sort=True):
        key = (key,) if not isinstance(key, tuple) else key
        error = group["eta_residual_deg"]
        records.append(dict(zip(groups, key), samples=len(group),
                            circular_bias_deg=circular_bias(error),
                            circular_std_deg=circular_std(error),
                            mae_deg=error.abs().mean(),
                            rmse_deg=np.sqrt(np.mean(error**2)),
                            p90_absolute_error_deg=np.quantile(error.abs(), .9,
                                                                method="inverted_cdf")))
    return pd.DataFrame.from_records(records)


def plot_by_height(summary: pd.DataFrame) -> None:
    tags = sorted(summary["tag"].unique())
    heights = sorted(summary["height_difference_m"].unique())
    x = np.arange(len(tags))
    width = .23
    colors = {0.: "#1565c0", 2.: "#00897b", 4.: "#8e24aa"}
    figure, axis = plt.subplots(figsize=(8.3, 3.8), layout="constrained")
    for index, height in enumerate(heights):
        values = summary[summary.height_difference_m.eq(height)].set_index("tag").loc[tags, "mae_deg"]
        axis.bar(x + (index - 1) * width, values, width=width,
                 color=colors[height], label=fr"$h={height:g}\,\mathrm{{m}}$")
    axis.set_xticks(x, tags)
    axis.set_ylabel("Raw PDoA residual MAE [deg]")
    axis.set_xlabel("Tag / deployment bearing")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18))
    figure.savefig(PLOTS_DIRECTORY / "pdoa_mae_by_tag_height.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_by_rotation(summary: pd.DataFrame) -> None:
    tags = sorted(summary["tag"].unique())
    rotations = sorted(summary["tag_rotation_deg"].unique())
    x = np.arange(len(tags))
    colors = {0.: "#1565c0", 90.: "#00897b", 180.: "#f57c00", 270.: "#8e24aa"}
    figure, axis = plt.subplots(figsize=(8.3, 3.8), layout="constrained")
    for rotation in rotations:
        values = summary[summary.tag_rotation_deg.eq(rotation)].set_index("tag").loc[tags, "mae_deg"]
        axis.plot(x, values, marker="o", markersize=4, linewidth=1.5,
                  color=colors[rotation], label=fr"$\psi={rotation:g}\degree$")
    axis.set_xticks(x, tags)
    axis.set_ylabel("Raw PDoA residual MAE [deg]")
    axis.set_xlabel("Tag / deployment bearing")
    axis.grid(alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18))
    figure.savefig(PLOTS_DIRECTORY / "pdoa_mae_by_tag_rotation.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    CSV_DIRECTORY.mkdir(exist_ok=True)
    PLOTS_DIRECTORY.mkdir(exist_ok=True)
    frame = pd.read_csv(DATASET)
    frame["height_difference_m"] = frame["height_difference_m"].astype(float)
    frame["tag_rotation_deg"] = frame["tag_rotation_deg"].astype(float)
    by_height = summarize(frame, ["tag", "gt_bearing_deg", "height_difference_m"])
    by_rotation = summarize(frame, ["tag", "gt_bearing_deg", "tag_rotation_deg"])
    by_height.to_csv(CSV_DIRECTORY / "pdoa_raw_by_tag_height.csv", index=False)
    by_rotation.to_csv(CSV_DIRECTORY / "pdoa_raw_by_tag_rotation.csv", index=False)
    plot_by_height(by_height)
    plot_by_rotation(by_rotation)
    print(f"Wrote {CSV_DIRECTORY / 'pdoa_raw_by_tag_height.csv'}")
    print(f"Wrote {CSV_DIRECTORY / 'pdoa_raw_by_tag_rotation.csv'}")
    print(f"Wrote {PLOTS_DIRECTORY / 'pdoa_mae_by_tag_height.pdf'}")
    print(f"Wrote {PLOTS_DIRECTORY / 'pdoa_mae_by_tag_rotation.pdf'}")


if __name__ == "__main__":
    main()
