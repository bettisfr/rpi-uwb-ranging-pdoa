#!/usr/bin/env python3
"""Audit alignment, RTK quality, refresh, and motion in the UAV UWB logs."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "uav-20260903"
OUTPUT = ROOT / "analysis" / "csv" / "uav_audit_summary.csv"


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated percentile for a nonempty numeric list."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    """Compact stable summary for nonempty measurement distributions."""
    return dict(median=median(values), p90=percentile(values, .9), mean=fmean(values))


def main() -> None:
    logs = sorted(SOURCE.glob("drone_*.csv"))
    if not logs:
        raise FileNotFoundError(f"No aligned UAV logs found under {SOURCE}")
    rows_by_series: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    rows_by_flight: dict[str, list[dict[str, str]]] = defaultdict(list)
    statuses, tags, ages = Counter(), Counter(), []
    total = 0
    for log in logs:
        with log.open(newline="") as source:
            for row in csv.DictReader(source):
                total += 1
                statuses[row["rtk_status"]] += 1
                tags[row["tag"]] += 1
                ages.append(float(row["uwb_age_s"]))
                rows_by_series[(log.name, row["tag"])].append(row)
                rows_by_flight[log.name].append(row)

    intervals_ms, step_m = [], []
    speeds_mps, alignment_motion_m = [], []
    for flight, flight_rows in rows_by_flight.items():
        # Several UWB rows share one flight-controller timestamp; retain one
        # RTK point per timestamp before differentiating the trajectory.
        points = {}
        for row in flight_rows:
            if row["rtk_status"] == "50":
                points.setdefault(int(row["fc_timestamp_ms"]), row)
        ordered = sorted(points.items())
        speed_by_timestamp = {}
        for (previous_time, previous), (current_time, current) in zip(ordered, ordered[1:]):
            elapsed_s = (current_time - previous_time) / 1000.
            if elapsed_s <= 0:
                continue
            speed = math.hypot(
                float(current["gt_x_m"]) - float(previous["gt_x_m"]),
                float(current["gt_y_m"]) - float(previous["gt_y_m"]),
            ) / elapsed_s
            speed_by_timestamp[current_time] = speed
        for row in flight_rows:
            timestamp = int(row["fc_timestamp_ms"])
            if row["rtk_status"] == "50" and timestamp in speed_by_timestamp:
                speed = speed_by_timestamp[timestamp]
                speeds_mps.append(speed)
                alignment_motion_m.append(speed * float(row["uwb_age_s"]))
    windows: dict[int, list[tuple[float, float]]] = {10: [], 20: [], 30: []}
    fixed_series = 0
    for (_, _), series in rows_by_series.items():
        series.sort(key=lambda row: int(row["fc_timestamp_ms"]))
        fixed = [row for row in series if row["rtk_status"] == "50"]
        if len(fixed) < 2:
            continue
        fixed_series += 1
        for previous, current in zip(fixed, fixed[1:]):
            elapsed = int(current["fc_timestamp_ms"]) - int(previous["fc_timestamp_ms"])
            if elapsed <= 0:
                continue
            intervals_ms.append(float(elapsed))
            step_m.append(math.hypot(
                float(current["gt_x_m"]) - float(previous["gt_x_m"]),
                float(current["gt_y_m"]) - float(previous["gt_y_m"]),
            ))
        for size in windows:
            for start in range(0, len(fixed) - size + 1, size):
                sample = fixed[start : start + size]
                elapsed_s = (int(sample[-1]["fc_timestamp_ms"]) - int(sample[0]["fc_timestamp_ms"])) / 1000.
                path = sum(math.hypot(
                    float(right["gt_x_m"]) - float(left["gt_x_m"]),
                    float(right["gt_y_m"]) - float(left["gt_y_m"]),
                ) for left, right in zip(sample, sample[1:]))
                windows[size].append((elapsed_s, path))

    summary = [
        dict(metric="rows", count=total, median="", p90="", mean=""),
        dict(metric="RTK fixed (status 50)", count=statuses["50"], median="", p90="", mean=""),
        dict(metric="RTK non-fixed (status 16)", count=statuses["16"], median="", p90="", mean=""),
        dict(metric="UWB alignment age [s]", count=len(ages), **describe(ages)),
        dict(metric="same-tag interval [ms], RTK fixed", count=len(intervals_ms), **describe(intervals_ms)),
        dict(metric="same-tag RTK step [m], RTK fixed", count=len(step_m), **describe(step_m)),
        dict(metric="RTK horizontal speed [m/s], RTK fixed", count=len(speeds_mps), **describe(speeds_mps)),
        dict(metric="motion explainable by UWB alignment age [m], RTK fixed", count=len(alignment_motion_m), **describe(alignment_motion_m)),
    ]
    for threshold_m in (.5, 1.):
        count = sum(value > threshold_m for value in alignment_motion_m)
        summary.append(dict(
            metric=f"alignment-motion bound > {threshold_m:g} m, RTK fixed",
            count=count, median="", p90="", mean=count / len(alignment_motion_m),
        ))
    for size, values in windows.items():
        duration, path = map(list, zip(*values))
        summary.extend((
            dict(metric=f"K={size} window duration [s], RTK fixed", count=len(duration), **describe(duration)),
            dict(metric=f"K={size} window path length [m], RTK fixed", count=len(path), **describe(path)),
        ))
        for threshold_m in (.25, .5, 1.):
            accepted = sum(value <= threshold_m for value in path)
            summary.append(dict(
                metric=f"K={size} windows with path <= {threshold_m:g} m, RTK fixed",
                count=accepted, median="", p90="", mean=accepted / len(path),
            ))
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=("metric", "count", "median", "p90", "mean"))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Aligned rows: {total}")
    print(f"RTK statuses: {dict(sorted(statuses.items()))}")
    print(f"Per-tag rows: {dict(sorted(tags.items()))}")
    print(f"Fixed same-tag interval: {describe(intervals_ms)} ms")
    print(f"Fixed same-tag RTK step: {describe(step_m)} m")
    print(f"RTK horizontal speed: {describe(speeds_mps)} m/s")
    print(f"Motion explainable by alignment age: {describe(alignment_motion_m)} m")
    for threshold_m in (.5, 1.):
        print(
            f"  alignment-motion bound > {threshold_m:g} m: "
            f"{sum(value > threshold_m for value in alignment_motion_m) / len(alignment_motion_m):.2%}"
        )
    for size, values in windows.items():
        duration, path = map(list, zip(*values))
        print(f"K={size}: duration={describe(duration)} s; path={describe(path)} m")
        for threshold_m in (.25, .5, 1.):
            print(
                f"  path <= {threshold_m:g} m: "
                f"{sum(value <= threshold_m for value in path) / len(path):.1%}"
            )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
