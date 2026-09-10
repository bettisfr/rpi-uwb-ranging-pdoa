#!/usr/bin/env python3
"""Visualize leave-one-distance-out range-correction results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis" / "outputs" / "distance_lodo"
PLOTS = ROOT / "analysis" / "plots"

LABELS = {
    "raw": "Raw",
    "constant_offset": "Offset",
    "linear_regression": "Linear",
    "quadratic_regression": "Quadratic",
    "random_forest": "Random Forest",
    "svr": "SVR",
    "mlp": "MLP",
}
COLORS = {
    "raw": "#616161",
    "constant_offset": "#1565c0",
    "linear_regression": "#00897b",
    "quadratic_regression": "#8e24aa",
    "random_forest": "#e53935",
    "svr": "#ef6c00",
    "mlp": "#6d4c41",
}
CANDIDATES = ["raw", "constant_offset", "linear_regression", "quadratic_regression"]
NONPARAMETRIC = ["random_forest", "svr", "mlp"]


def horizontal(axis, metrics: pd.DataFrame, methods: list[str], metric: str, title: str) -> None:
    selected = metrics.set_index("method").loc[methods]
    y = np.arange(len(methods))
    axis.barh(y, selected[metric] * 100, color=[COLORS[name] for name in methods], height=.62)
    axis.set_yticks(y, [LABELS[name] for name in methods])
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("Error [cm]")
    axis.grid(axis="x", alpha=.25)
    axis.set_axisbelow(True)


def overall_plot(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 5.4), layout="constrained")
    horizontal(axes[0, 0], metrics, CANDIDATES, "mae_m", "MAE: calibration candidates")
    horizontal(axes[0, 1], metrics, NONPARAMETRIC, "mae_m", "MAE: nonlinear baselines")
    horizontal(axes[1, 0], metrics, CANDIDATES, "rmse_m", "RMSE: calibration candidates")
    horizontal(axes[1, 1], metrics, NONPARAMETRIC, "rmse_m", "RMSE: nonlinear baselines")
    figure.savefig(PLOTS / "range_lodo_overall.pdf", bbox_inches="tight")
    plt.close(figure)


def line_panel(axis, metrics: pd.DataFrame, methods: list[str], title: str) -> None:
    for method in methods:
        selected = metrics[metrics.method.eq(method)].sort_values("distance_m")
        axis.plot(selected.distance_m, selected.mae_m * 100, marker="o", markersize=3.5,
                  linewidth=1.5, color=COLORS[method], label=LABELS[method])
    axis.set_title(title)
    axis.set_xlabel("Held-out ground distance [m]")
    axis.set_ylabel("MAE [cm]")
    axis.set_xticks(np.arange(2, 31, 4))
    axis.grid(alpha=.25)
    axis.legend(frameon=False, loc="upper left")


def by_distance_plot(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout="constrained")
    line_panel(axes[0], metrics, CANDIDATES, "Simple corrections")
    line_panel(axes[1], metrics, NONPARAMETRIC, "Nonlinear baselines")
    figure.savefig(PLOTS / "range_lodo_by_distance.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    overall_plot(pd.read_csv(OUTPUT / "overall_metrics.csv"))
    by_distance_plot(pd.read_csv(OUTPUT / "metrics_by_distance.csv"))
    print(f"Wrote {PLOTS / 'range_lodo_overall.pdf'}")
    print(f"Wrote {PLOTS / 'range_lodo_by_distance.pdf'}")


if __name__ == "__main__":
    main()
