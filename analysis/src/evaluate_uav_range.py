#!/usr/bin/env python3
"""Evaluate raw and static-calibrated UWB range against RTK-referenced UAV geometry.

The static quadratic correction is fitted only on the static development pool
and is never adapted to the UAV records.  This first dynamic benchmark is
range-only: it intentionally does not require drone yaw or a PDoA convention.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from workflow.distance_models import make_model
from workflow.offset import load_assigned_rows
from recalculate_uav_local_frame import local_xy


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "analysis" / "csv" / "static_measurements.csv"
UAV_DIRECTORY = ROOT / "data" / "uav-20260903"
CONFIG = ROOT / "analysis" / "configs" / "distance_models.json"
OUTPUT = ROOT / "analysis" / "outputs" / "uav_range_transfer"
# The DJI height is measured from ground level.  The DWM1002 antenna is
# mounted 55 cm above the aircraft reference, whereas each DWM1001 tag is
# mounted 45 cm above ground on a field stake.
NODE_ANTENNA_OFFSET_M = .55
TAG_HEIGHT_M = .45
MAX_ALIGNMENT_AGE_S = .10
MAX_INTERPOLATION_GAP_S = .50
TAG_LAYOUT = {
    "dw00": (15., 28.), "dw01": (35., 18.), "dw02": (55., 10.),
    "dw03": (75., 24.), "dw04": (90., 30.), "dw05": (105., 24.),
    "dw06": (125., 10.), "dw07": (145., 18.), "dw08": (165., 28.),
}


def metric(values: list[float]) -> dict[str, float]:
    """Return signed range-error metrics in metres."""
    array = np.asarray(values)
    return dict(
        samples=len(array), bias_m=float(array.mean()), mae_m=float(np.abs(array).mean()),
        rmse_m=float(np.sqrt(np.mean(array ** 2))),
        p90_absolute_error_m=float(np.quantile(np.abs(array), .9, method="inverted_cdf")),
    )


def tag_xy(tag: str) -> tuple[float, float]:
    """Return a surveyed tag coordinate in the O/Q4 local frame."""
    bearing_deg, radius_m = TAG_LAYOUT[tag]
    bearing = math.radians(bearing_deg)
    return radius_m * math.cos(bearing), radius_m * math.sin(bearing)


def timestamp(value: str) -> datetime:
    """Parse both ISO spellings emitted by the flight and UWB loggers."""
    return datetime.fromisoformat(value)


def rtk_xy(row: dict[str, str], reference: dict[str, float]) -> tuple[float, float]:
    """Reconstruct local coordinates from the recorded RTK fix offline."""
    return local_xy(reference, float(row["rtk_lat_deg"]), float(row["rtk_lon_deg"]))


def interpolate_rtk(
    points: list[tuple[datetime, dict[str, str]]],
    target: datetime,
    reference: dict[str, float],
) -> tuple[float, float, float, float] | None:
    """Linearly interpolate a fixed RTK point at an UWB timestamp.

    The return contains x, y, height, and bracketing time gap.  Records outside
    a fixed--fixed interval or across a large temporal gap are not interpolated.
    """
    times = [point[0] for point in points]
    position = bisect_left(times, target)
    if position == 0 or position == len(points):
        return None
    left_time, left = points[position - 1]
    right_time, right = points[position]
    gap_s = (right_time - left_time).total_seconds()
    if gap_s <= 0 or gap_s > MAX_INTERPOLATION_GAP_S:
        return None
    fraction = (target - left_time).total_seconds() / gap_s
    if not 0 <= fraction <= 1:
        return None
    left_x, left_y = rtk_xy(left, reference)
    right_x, right_y = rtk_xy(right, reference)

    def blend(field: str) -> float:
        return float(left[field]) + fraction * (float(right[field]) - float(left[field]))
    return (
        left_x + fraction * (right_x - left_x),
        left_y + fraction * (right_y - left_y),
        blend("height_fusion_m"),
        gap_s,
    )


def main() -> None:
    with CONFIG.open() as source:
        config = json.load(source)
    static_rows, _, static_digest = load_assigned_rows(STATIC, ROOT / "analysis" / "outputs")
    development = [row for row in static_rows if row["subset"] in ("train", "validation") and row["quality_enabled"]]
    model = make_model("quadratic_regression", config).fit(
        np.asarray([float(row["estimated_ground_distance_m"]) for row in development]).reshape(-1, 1),
        np.asarray([float(row["target_ground_distance_m"]) for row in development]),
    )
    with (UAV_DIRECTORY / "reference.json").open() as source:
        reference = json.load(source)

    records = []
    excluded = defaultdict(int)
    for path in sorted(UAV_DIRECTORY.glob("drone_*.csv")):
        with path.open(newline="") as source:
            source_rows = list(csv.DictReader(source))
        # One RTK state per flight-controller timestamp is sufficient for
        # interpolation; every UWB event may share that state.
        state_by_time = {}
        for row in source_rows:
            if row["rtk_status"] == "50":
                state_by_time.setdefault(timestamp(row["time"]), row)
        fixed_states = sorted(state_by_time.items())
        speed_by_timestamp = {}
        for (_, left), (_, right) in zip(fixed_states, fixed_states[1:]):
            elapsed_s = (int(right["fc_timestamp_ms"]) - int(left["fc_timestamp_ms"])) / 1000.
            if elapsed_s > 0:
                left_x, left_y = rtk_xy(left, reference)
                right_x, right_y = rtk_xy(right, reference)
                speed_by_timestamp[int(right["fc_timestamp_ms"])] = math.hypot(
                    right_x - left_x, right_y - left_y,
                ) / elapsed_s
        for row in source_rows:
            if row["rtk_status"] != "50":
                excluded["rtk_non_fixed"] += 1
                continue
            if float(row["uwb_age_s"]) > MAX_ALIGNMENT_AGE_S:
                excluded["alignment_age"] += 1
                continue
            tag = row["tag"]
            if tag not in TAG_LAYOUT:
                excluded["unknown_tag"] += 1
                continue
            interpolated = interpolate_rtk(fixed_states, timestamp(row["uwb_time"]), reference)
            if interpolated is None:
                excluded["cannot_interpolate_rtk"] += 1
                continue
            node_x, node_y, node_height, interpolation_gap_s = interpolated
            height_difference = node_height + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
            if height_difference < 0:
                excluded["invalid_height"] += 1
                continue
            nearest_x, nearest_y = rtk_xy(row, reference)
            nearest_height_difference = (
                float(row["height_fusion_m"]) + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
            )
            if nearest_height_difference < 0:
                excluded["invalid_nearest_height"] += 1
                continue
            tag_x, tag_y = tag_xy(tag)
            horizontal_truth = math.hypot(node_x - tag_x, node_y - tag_y)
            true_range = math.hypot(horizontal_truth, height_difference)
            range_m = float(row["range_cm"]) / 100.
            if range_m < height_difference or range_m < nearest_height_difference:
                excluded["invalid_horizontal_projection"] += 1
                continue
            raw_horizontal = math.sqrt(range_m ** 2 - height_difference ** 2)
            corrected_horizontal = float(model.predict(np.asarray([[raw_horizontal]]))[0])
            nearest_horizontal_truth = math.hypot(nearest_x - tag_x, nearest_y - tag_y)
            nearest_true_range = math.hypot(nearest_horizontal_truth, nearest_height_difference)
            nearest_raw_horizontal = math.sqrt(range_m ** 2 - nearest_height_difference ** 2)
            nearest_corrected_horizontal = float(model.predict(np.asarray([[nearest_raw_horizontal]]))[0])
            records.append(dict(
                    flight=path.stem, tag=tag, rtk_status=50, uwb_age_s=float(row["uwb_age_s"]),
                    interpolation_gap_s=interpolation_gap_s, node_x_m=node_x, node_y_m=node_y,
                    node_height_m=node_height, tag_x_m=tag_x, tag_y_m=tag_y,
                    nearest_node_x_m=nearest_x, nearest_node_y_m=nearest_y,
                    nearest_horizontal_truth_m=nearest_horizontal_truth,
                    speed_mps=speed_by_timestamp.get(int(row["fc_timestamp_ms"]), float("nan")),
                    yaw_deg=float(row["yaw_deg"]), roll_deg=float(row["roll_deg"]), pitch_deg=float(row["pitch_deg"]),
                    pdoa_deg=float(row["pdoa_deg"]),
                    firmware_x_m=float(row["x_cm"]) / 100., firmware_y_m=float(row["y_cm"]) / 100.,
                    horizontal_truth_m=horizontal_truth, true_range_m=true_range,
                    raw_range_m=range_m, raw_horizontal_m=raw_horizontal,
                    corrected_horizontal_m=corrected_horizontal,
                    raw_range_error_m=range_m - true_range,
                    raw_horizontal_error_m=raw_horizontal - horizontal_truth,
                    corrected_horizontal_error_m=corrected_horizontal - horizontal_truth,
                    nearest_raw_range_error_m=range_m - nearest_true_range,
                    nearest_raw_horizontal_error_m=nearest_raw_horizontal - nearest_horizontal_truth,
                    nearest_corrected_horizontal_error_m=nearest_corrected_horizontal - nearest_horizontal_truth,
                    nearest_corrected_horizontal_m=nearest_corrected_horizontal,
                ))
    if not records:
        raise ValueError("No RTK-fixed, tightly aligned UAV measurements")
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "predictions.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    groups = [("all", records)]
    groups.extend((tag, [row for row in records if row["tag"] == tag]) for tag in sorted(TAG_LAYOUT))
    groups.extend((flight, [row for row in records if row["flight"] == flight])
                  for flight in sorted({row["flight"] for row in records}))
    for label, selected in groups:
        for name, field in (("raw_3d", "raw_range_error_m"),
                            ("raw_horizontal", "raw_horizontal_error_m"),
                            ("static_quadratic_horizontal", "corrected_horizontal_error_m"),
                            ("nearest_raw_3d", "nearest_raw_range_error_m"),
                            ("nearest_raw_horizontal", "nearest_raw_horizontal_error_m"),
                            ("nearest_static_quadratic_horizontal", "nearest_corrected_horizontal_error_m")):
            summary.append(dict(group=label, method=name, **metric([row[field] for row in selected])))
    with (OUTPUT / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", static_source_sha256=static_digest,
        transfer="Static quadratic range correction fitted on static train+validation only; no UAV fitting",
        selected_uav_records=len(records), excluded=dict(excluded),
        rtk_policy="status 50 only, linearly interpolated at each UWB timestamp",
        max_alignment_age_s=MAX_ALIGNMENT_AGE_S, max_interpolation_gap_s=MAX_INTERPOLATION_GAP_S,
        vertical_reference=(
            f"height_fusion_m + node antenna offset {NODE_ANTENNA_OFFSET_M:g} m "
            f"- fixed tag height {TAG_HEIGHT_M:g} m"
        ),
        local_geometry="UAV RTK local O/Q4 coordinates and surveyed fixed tag layout",
    ), indent=2) + "\n")
    for row in summary:
        if row["group"] == "all":
            print(f"{row['method']}: MAE={row['mae_m']:.3f} m, RMSE={row['rmse_m']:.3f} m, P90={row['p90_absolute_error_m']:.3f} m")
    print(f"Selected {len(records)} records; excluded {dict(excluded)}")


if __name__ == "__main__":
    main()
