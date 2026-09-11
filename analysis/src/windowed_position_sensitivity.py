#!/usr/bin/env python3
"""Grouped validation/test coverage--accuracy sensitivity for temporal windows.

The range and PDoA corrections are fitted on grouped training configurations
only.  Consecutive observations are then aggregated within one tag/run, and
the circular PDoA standard deviation acts solely as an observable confidence
gate.  No tag, bearing, rotation, or ground truth is an inference input.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from workflow.distance_models import make_model
from workflow.offset import load_assigned_rows
from workflow.phase import wrap
from workflow.phase_trigonometric import fit_trigonometric, predict_correction
from workflow.position_benchmark import metrics, reconstruct, write_csv


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "analysis" / "csv" / "static_measurements.csv"
CONFIG = ROOT / "analysis" / "configs" / "distance_models.json"
OUTPUT = ROOT / "analysis" / "outputs" / "windowed_position_sensitivity"
WINDOW_SIZES = (10, 20, 30)
GATES_DEG = (5., 10., 15., 20., 25., 30., 40., 50., 60.)


def circular_mean(values: list[float]) -> float:
    """Return the circular mean of degree-valued observations."""
    return math.degrees(math.atan2(
        sum(math.sin(math.radians(value)) for value in values),
        sum(math.cos(math.radians(value)) for value in values),
    ))


def circular_std(values: list[float]) -> float:
    """Return circular standard deviation in degrees."""
    sine = np.mean(np.sin(np.deg2rad(values)))
    cosine = np.mean(np.cos(np.deg2rad(values)))
    resultant = max(float(np.hypot(sine, cosine)), 1e-15)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant))))


def aggregate(rows: list[dict], corrected_range: dict[int, float], corrected_phase: dict[int, float],
              window_size: int) -> list[dict]:
    """Aggregate only complete non-overlapping windows in a single subset."""
    series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        series[(row["run_file"], row["tag"])].append(row)
    records = []
    for (_, tag), observations in series.items():
        observations.sort(key=lambda row: int(row["sequence"]))
        for start in range(0, len(observations) - window_size + 1, window_size):
            window = observations[start : start + window_size]
            lines = [int(row["csv_line"]) for row in window]
            if any(line not in corrected_range or line not in corrected_phase for line in lines):
                continue
            first = window[0]
            raw_rho = [float(row["estimated_ground_distance_m"]) for row in window]
            if not all(math.isfinite(value) for value in raw_rho):
                continue
            records.append(dict(
                subset=first["subset"], tag=tag, window_size=window_size,
                height_difference_m=float(first["height_difference_m"]),
                x_true_m=float(first["gt_x_m"]), y_true_m=float(first["gt_y_m"]),
                pdoa_std_deg=circular_std([float(row["pdoa_deg"]) for row in window]),
                raw_rho_m=float(np.mean(raw_rho)),
                corrected_rho_m=float(np.mean([corrected_range[line] for line in lines])),
                raw_phase_deg=circular_mean([float(row["pdoa_deg"]) for row in window]),
                corrected_phase_deg=circular_mean([corrected_phase[line] for line in lines]),
            ))
    return records


def evaluate(records: list[dict], gate: float | None) -> dict:
    """Evaluate global range/PDoA reconstruction under one confidence gate."""
    stable = np.asarray([row["pdoa_std_deg"] for row in records])
    selected = np.ones(len(records), dtype=bool) if gate is None else stable <= gate
    chosen = [row for row, keep in zip(records, selected) if keep]
    if not chosen:
        return dict(accepted_windows=0, accepted_fraction=0.0)
    height = np.asarray([row["height_difference_m"] for row in chosen])
    x_hat, y_hat, clipped = reconstruct(
        [row["corrected_rho_m"] for row in chosen], height,
        [row["corrected_phase_deg"] for row in chosen],
    )
    return dict(
        accepted_windows=len(chosen), accepted_fraction=float(selected.mean()),
        **metrics(
            np.asarray([row["x_true_m"] for row in chosen]),
            np.asarray([row["y_true_m"] for row in chosen]), x_hat, y_hat, clipped,
        ),
    )


def main() -> None:
    with CONFIG.open() as source:
        config = json.load(source)
    rows, split, digest = load_assigned_rows(DATASET, ROOT / "analysis" / "outputs")
    train = [row for row in rows if row["subset"] == "train" and row["quality_enabled"]]
    phase_train = [row for row in rows if row["subset"] == "train"]
    distance_model = make_model("quadratic_regression", config).fit(
        np.asarray([float(row["estimated_ground_distance_m"]) for row in train]).reshape(-1, 1),
        np.asarray([float(row["target_ground_distance_m"]) for row in train]),
    )
    phase_model = fit_trigonometric(
        np.asarray([float(row["pdoa_deg"]) for row in phase_train]),
        np.asarray([float(row["eta_expected_deg"]) for row in phase_train]),
        np.full(len(phase_train), "train", dtype=object),
    )
    corrected_range: dict[int, float] = {}
    eligible = [row for row in rows if row["quality_enabled"]]
    predictions = distance_model.predict(
        np.asarray([float(row["estimated_ground_distance_m"]) for row in eligible]).reshape(-1, 1)
    )
    for row, value in zip(eligible, predictions):
        corrected_range[int(row["csv_line"])] = float(value)
    phase_values = np.asarray([float(row["pdoa_deg"]) for row in rows])
    correction, _, _ = predict_correction(phase_values, phase_model)
    corrected_phase = {int(row["csv_line"]): float(wrap(phase - bias))
                       for row, phase, bias in zip(rows, phase_values, correction)}

    all_records, summary = [], []
    for window_size in WINDOW_SIZES:
        for subset in ("validation", "test"):
            records = aggregate([row for row in rows if row["subset"] == subset], corrected_range,
                                corrected_phase, window_size)
            all_records.extend(records)
            for gate in (None, *GATES_DEG):
                summary.append(dict(
                    subset=subset, window_size=window_size,
                    pdoa_std_gate_deg="all" if gate is None else gate,
                    **evaluate(records, gate),
                ))
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "window_records.csv", all_records)
    write_csv(OUTPUT / "coverage_accuracy.csv", summary)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", source_sha256=digest, split=split,
        protocol="grouped train/validation/test; correction fitted on train only",
        window_sizes=list(WINDOW_SIZES), gate_thresholds_deg=list(GATES_DEG),
        aggregation="non-overlapping consecutive windows; arithmetic range mean and circular PDoA mean",
        allowed_inputs=["range_m", "pdoa_deg", "height_difference_m", "PDoA within-window dispersion"],
        forbidden_inputs=["tag", "bearing", "tag_rotation_deg", "ground-truth position"],
    ), indent=2) + "\n")
    for row in summary:
        if row["pdoa_std_gate_deg"] in ("all", 20.0):
            print(
                f"{row['subset']} K={row['window_size']} gate={row['pdoa_std_gate_deg']}: "
                f"coverage={row['accepted_fraction']:.1%}, MAE={row['position_mae_m']:.3f} m, "
                f"P90={row['position_p90_m']:.3f} m"
            )


if __name__ == "__main__":
    main()
