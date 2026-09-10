#!/usr/bin/env python3
"""Measure short-term stability from consecutive static observations only.

The per-window metrics deliberately use no ground-truth coordinates, bearing,
or distance.  Ground-truth fields are retained only to stratify the resulting
diagnostic table after the fact.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "analysis" / "csv" / "static_measurements.csv"
CSV_DIRECTORY = ROOT / "analysis" / "csv"


def circular_standard_deviation_deg(values: list[float]) -> float:
    """Return circular standard deviation, in degrees."""
    sine = fmean(math.sin(math.radians(value)) for value in values)
    cosine = fmean(math.cos(math.radians(value)) for value in values)
    resultant = math.hypot(sine, cosine)
    return math.degrees(math.sqrt(-2.0 * math.log(max(resultant, 1e-15))))


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a nonempty sequence."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def wrap_degrees(value: float) -> float:
    """Map an angle to [-180, 180)."""
    return (value + 180.0) % 360.0 - 180.0


def pearson(first: list[float], second: list[float]) -> float:
    """Return Pearson's correlation coefficient for two equally sized lists."""
    first_mean = fmean(first)
    second_mean = fmean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    first_scale = math.sqrt(sum((x - first_mean) ** 2 for x in first))
    second_scale = math.sqrt(sum((y - second_mean) ** 2 for y in second))
    return numerator / (first_scale * second_scale)


def main() -> None:
    with DATASET.open(newline="") as source:
        rows = list(csv.DictReader(source))

    series: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        series[(row["run_file"], row["tag"])].append(row)

    windows: list[dict[str, object]] = []
    for window_size in (20, 30):
        for (_, tag), observations in series.items():
            observations.sort(key=lambda row: int(row["sequence"]))
            for start in range(0, len(observations) - window_size + 1, window_size):
                sample = observations[start : start + window_size]
                ranges = [float(row["range_m"]) for row in sample]
                phases = [float(row["pdoa_deg"]) for row in sample]
                first = sample[0]
                range_mean = fmean(ranges)
                phase_mean = math.degrees(
                    math.atan2(
                        fmean(math.sin(math.radians(value)) for value in phases),
                        fmean(math.cos(math.radians(value)) for value in phases),
                    )
                )
                height = float(first["height_difference_m"])
                horizontal_range = math.sqrt(max(0.0, range_mean**2 - height**2))
                windows.append(
                    {
                        "window_size": window_size,
                        "run_file": first["run_file"],
                        "tag": tag,
                        "height_difference_m": first["height_difference_m"],
                        "target_ground_distance_m": first["target_ground_distance_m"],
                        "tag_rotation_deg": first["tag_rotation_deg"],
                        "gt_bearing_deg": first["gt_bearing_deg"],
                        "range_mean_m": range_mean,
                        "range_std_cm": 100.0 * stdev(ranges),
                        "pdoa_circular_std_deg": circular_standard_deviation_deg(phases),
                        "range_abs_error_cm": 100.0 * abs(
                            horizontal_range - float(first["target_ground_distance_m"])
                        ),
                        "pdoa_abs_residual_deg": abs(
                            wrap_degrees(phase_mean - float(first["eta_expected_deg"]))
                        ),
                    }
                )

    window_path = CSV_DIRECTORY / "static_consecutive_window_stability.csv"
    with window_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)

    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in windows:
        grouped[(int(row["window_size"]), str(row["tag"]))].append(row)

    summary: list[dict[str, object]] = []
    for (window_size, tag), values in sorted(grouped.items()):
        range_std = [float(row["range_std_cm"]) for row in values]
        pdoa_std = [float(row["pdoa_circular_std_deg"]) for row in values]
        summary.append(
            {
                "window_size": window_size,
                "tag": tag,
                "window_count": len(values),
                "range_std_median_cm": percentile(range_std, 0.5),
                "range_std_p90_cm": percentile(range_std, 0.9),
                "pdoa_std_median_deg": percentile(pdoa_std, 0.5),
                "pdoa_std_p90_deg": percentile(pdoa_std, 0.9),
            }
        )

    summary_path = CSV_DIRECTORY / "static_consecutive_window_stability_by_tag.csv"
    with summary_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {window_path}")
    print(f"Wrote {summary_path}")
    for row in summary:
        if row["window_size"] == 20:
            print(
                f"{row['tag']}: range median/p90 "
                f"{row['range_std_median_cm']:.2f}/{row['range_std_p90_cm']:.2f} cm; "
                f"PDoA median/p90 {row['pdoa_std_median_deg']:.2f}/"
                f"{row['pdoa_std_p90_deg']:.2f} deg"
            )

    for window_size in (20, 30):
        subset = [row for row in windows if row["window_size"] == window_size]
        stability = [float(row["pdoa_circular_std_deg"]) for row in subset]
        residual = [float(row["pdoa_abs_residual_deg"]) for row in subset]
        print(
            f"window={window_size}: corr(PDoA std, absolute PDoA residual)="
            f"{pearson(stability, residual):.3f}"
        )
        for threshold in (10.0, 20.0, 40.0):
            accepted = [row for row in subset if float(row["pdoa_circular_std_deg"]) <= threshold]
            print(
                f"  std<={threshold:g} deg: {len(accepted) / len(subset):.1%} accepted, "
                f"PDoA MAE={fmean(float(row['pdoa_abs_residual_deg']) for row in accepted):.2f} deg"
            )


if __name__ == "__main__":
    main()
