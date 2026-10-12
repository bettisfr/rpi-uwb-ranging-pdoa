#!/usr/bin/env python3
"""Evaluate raw and static-calibrated UWB range against RTK-referenced UAV geometry.

The static quadratic correction is fitted only on the static development pool
and is never adapted to the UAV records.  This first dynamic benchmark is
range-only: it intentionally does not require drone yaw or a PDoA convention.

Ground-truth timing: the logger writes ~10 Hz rows but the RTK fix refreshes
at ~1 Hz and is held in between, so the reference is interpolated between fix
changes only.  A constant reporting latency is estimated per flight and each
flight is evaluated with the latency estimated on the other one.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from workflow.distance_models import make_model
from workflow.offset import load_assigned_rows
from uav.config import (
    ANALYSIS,
    MAX_ALIGNMENT_AGE_S,
    NODE_ANTENNA_OFFSET_M,
    ROOT,
    RTK_MAX_FIX_GAP_S,
    TAG_HEIGHT_M,
    TAG_LAYOUT,
    UAV_DATA,
    tag_xy,
)
from uav.coordinates import local_xy


STATIC = ANALYSIS / "csv" / "static_measurements.csv"
UAV_DIRECTORY = UAV_DATA
CONFIG = ANALYSIS / "configs" / "distance_models.json"
OUTPUT = ANALYSIS / "outputs" / "uav_range_transfer"
LATENCY_GRID_S = np.round(np.arange(0., 1. + 1e-9, .025), 3)


def metric(values: list[float]) -> dict[str, float]:
    """Return signed range-error metrics in metres."""
    array = np.asarray(values)
    return dict(
        samples=len(array), bias_m=float(array.mean()), mae_m=float(np.abs(array).mean()),
        rmse_m=float(np.sqrt(np.mean(array ** 2))),
        p90_absolute_error_m=float(np.quantile(np.abs(array), .9, method="inverted_cdf")),
    )


def timestamp(value: str) -> datetime:
    """Parse both ISO spellings emitted by the flight and UWB loggers."""
    return datetime.fromisoformat(value)


def rtk_xy(row: dict[str, str], reference: dict[str, float]) -> tuple[float, float]:
    """Reconstruct local coordinates from the recorded RTK fix offline."""
    return local_xy(reference, float(row["rtk_lat_deg"]), float(row["rtk_lon_deg"]))


def change_points(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mask of the rows where a held telemetry value is refreshed."""
    changed = np.ones(len(values), dtype=bool)
    changed[1:] = np.any(values[1:].reshape(len(values) - 1, -1) != values[:-1].reshape(len(values) - 1, -1), axis=1)
    return changed


def fix_track(rows: list[dict[str, str]], reference: dict[str, float]):
    """RTK-fixed change points of position and fused height, in seconds from the first fixed row."""
    fixed = sorted((row for row in rows if row["rtk_status"] == "50"), key=lambda row: timestamp(row["time"]))
    origin = timestamp(fixed[0]["time"])
    seconds = np.asarray([(timestamp(row["time"]) - origin).total_seconds() for row in fixed])
    latlon = np.asarray([(row["rtk_lat_deg"], row["rtk_lon_deg"]) for row in fixed])
    height = np.asarray([float(row["height_fusion_m"]) for row in fixed])
    position_change, height_change = change_points(seconds, latlon), change_points(seconds, height)
    xy = np.asarray([rtk_xy(row, reference) for row, keep in zip(fixed, position_change) if keep])
    rtk_h = np.asarray([float(row["rtk_h_m"]) for row, keep in zip(fixed, position_change) if keep])
    return dict(origin=origin, t_xy=seconds[position_change], xy=xy, rtk_h=rtk_h,
                t_h=seconds[height_change], h=height[height_change])


def _bracket(times: np.ndarray, target: np.ndarray, max_gap: float):
    right = np.clip(np.searchsorted(times, target), 1, len(times) - 1)
    left = right - 1
    gap = times[right] - times[left]
    fraction = np.where(gap > 0, (target - times[left]) / np.where(gap > 0, gap, 1.), 0.)
    valid = (target >= times[0]) & (target <= times[-1]) & (gap > 0) & (gap <= max_gap)
    return left, right, fraction, gap, valid


def track_at(track, seconds: np.ndarray):
    """Interpolated x, y, fused height, fix gap and segment speed; validity mask."""
    left, right, fraction, gap, valid_xy = _bracket(track["t_xy"], seconds, RTK_MAX_FIX_GAP_S)
    xy = track["xy"][left] + fraction[:, None] * (track["xy"][right] - track["xy"][left])
    speed = np.hypot(*(track["xy"][right] - track["xy"][left]).T) / np.where(gap > 0, gap, np.nan)
    # Fused height can legitimately stay constant (on ground, steady hover): no gap limit.
    h_left, h_right, h_fraction, _, valid_h = _bracket(track["t_h"], seconds, np.inf)
    height = track["h"][h_left] + h_fraction * (track["h"][h_right] - track["h"][h_left])
    return xy[:, 0], xy[:, 1], height, gap, speed, valid_xy & valid_h


def rtk_height_at(track, seconds: np.ndarray) -> np.ndarray:
    """Ellipsoidal RTK height, interpolated like x/y (same fix change points)."""
    left, right, fraction, _, _ = _bracket(track["t_xy"], seconds, RTK_MAX_FIX_GAP_S)
    return track["rtk_h"][left] + fraction * (track["rtk_h"][right] - track["rtk_h"][left])


def latency_objective(track, uwb_seconds, range_m, tags_xy) -> np.ndarray:
    """Std of the raw 3D range residual per candidate latency (bias-invariant)."""
    values = []
    for latency in LATENCY_GRID_S:
        x, y, height, _, _, valid = track_at(track, uwb_seconds + latency)
        true_range = np.hypot(np.hypot(x - tags_xy[:, 0], y - tags_xy[:, 1]),
                              height + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M)
        values.append(float(np.std((range_m - true_range)[valid])))
    return np.asarray(values)


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

    excluded = defaultdict(int)
    flights = {}
    for path in sorted(UAV_DIRECTORY.glob("drone_*.csv")):
        with path.open(newline="") as source:
            source_rows = list(csv.DictReader(source))
        track = fix_track(source_rows, reference)
        selected = []
        for row in source_rows:
            if row["rtk_status"] != "50":
                excluded["rtk_non_fixed"] += 1
            elif float(row["uwb_age_s"]) > MAX_ALIGNMENT_AGE_S:
                excluded["alignment_age"] += 1
            elif row["tag"] not in TAG_LAYOUT:
                excluded["unknown_tag"] += 1
            else:
                selected.append(row)
        seconds = np.asarray([(timestamp(row["uwb_time"]) - track["origin"]).total_seconds() for row in selected])
        range_m = np.asarray([float(row["range_cm"]) / 100. for row in selected])
        tags_xy = np.asarray([tag_xy(row["tag"]) for row in selected])
        objective = latency_objective(track, seconds, range_m, tags_xy)
        flights[path.stem] = dict(track=track, rows=selected, seconds=seconds, range_m=range_m, tags_xy=tags_xy,
                                  own_latency_s=float(LATENCY_GRID_S[np.argmin(objective)]),
                                  objective=objective.tolist())
    if len(flights) != 2:
        raise ValueError("Cross-flight latency transfer expects exactly two flights")

    records = []
    latency_used = {}
    for name, flight in flights.items():
        other = next(value for key, value in flights.items() if key != name)
        latency = other["own_latency_s"]
        latency_used[name] = latency
        x, y, height, gap, speed, valid = track_at(flight["track"], flight["seconds"] + latency)
        rtk_height = rtk_height_at(flight["track"], flight["seconds"] + latency)
        for i, row in enumerate(flight["rows"]):
            if not valid[i]:
                excluded["cannot_interpolate_rtk"] += 1
                continue
            height_difference = height[i] + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
            if height_difference < 0:
                excluded["invalid_height"] += 1
                continue
            range_m = flight["range_m"][i]
            if range_m < height_difference:
                excluded["invalid_horizontal_projection"] += 1
                continue
            tag_x, tag_y = flight["tags_xy"][i]
            horizontal_truth = float(np.hypot(x[i] - tag_x, y[i] - tag_y))
            true_range = float(np.hypot(horizontal_truth, height_difference))
            raw_horizontal = float(np.sqrt(range_m ** 2 - height_difference ** 2))
            corrected_horizontal = float(model.predict(np.asarray([[raw_horizontal]]))[0])
            records.append(dict(
                flight=name, tag=row["tag"], uwb_time=row["uwb_time"], rtk_status=50,
                uwb_age_s=float(row["uwb_age_s"]), latency_s=latency, interpolation_gap_s=float(gap[i]),
                node_x_m=float(x[i]), node_y_m=float(y[i]), node_height_m=float(height[i]),
                rtk_ellipsoidal_h_m=float(rtk_height[i]),
                tag_x_m=tag_x, tag_y_m=tag_y, speed_mps=float(speed[i]),
                yaw_deg=float(row["yaw_deg"]), roll_deg=float(row["roll_deg"]), pitch_deg=float(row["pitch_deg"]),
                pdoa_deg=float(row["pdoa_deg"]),
                firmware_x_m=float(row["x_cm"]) / 100., firmware_y_m=float(row["y_cm"]) / 100.,
                horizontal_truth_m=horizontal_truth, true_range_m=true_range,
                raw_range_m=range_m, raw_horizontal_m=raw_horizontal,
                corrected_horizontal_m=corrected_horizontal,
                raw_range_error_m=range_m - true_range,
                raw_horizontal_error_m=raw_horizontal - horizontal_truth,
                corrected_horizontal_error_m=corrected_horizontal - horizontal_truth,
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
    groups.extend((flight, [row for row in records if row["flight"] == flight]) for flight in sorted(flights))
    for label, selected in groups:
        for name, field in (("raw_3d", "raw_range_error_m"),
                            ("raw_horizontal", "raw_horizontal_error_m"),
                            ("static_quadratic_horizontal", "corrected_horizontal_error_m")):
            summary.append(dict(group=label, method=name, **metric([row[field] for row in selected])))
    with (OUTPUT / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", static_source_sha256=static_digest,
        transfer="Static quadratic range correction fitted on static train+validation only; no UAV fitting",
        selected_uav_records=len(records), excluded=dict(excluded),
        rtk_policy=("status 50 only; x/y and fused height interpolated between their own change points "
                    "(1 Hz RTK held in ~10 Hz rows)"),
        rtk_max_fix_gap_s=RTK_MAX_FIX_GAP_S, max_alignment_age_s=MAX_ALIGNMENT_AGE_S,
        latency=dict(grid_s=[float(LATENCY_GRID_S[0]), float(LATENCY_GRID_S[-1]), .025],
                     objective="std of raw 3D range residual",
                     own_estimate_s={name: flight["own_latency_s"] for name, flight in flights.items()},
                     applied_s=latency_used, rule="each flight uses the latency estimated on the other flight"),
        vertical_reference=(
            f"height_fusion_m + node antenna offset {NODE_ANTENNA_OFFSET_M:g} m "
            f"- fixed tag height {TAG_HEIGHT_M:g} m"
        ),
        local_geometry="UAV RTK local O/Q4 coordinates and surveyed fixed tag layout",
    ), indent=2) + "\n")
    for row in summary:
        if row["group"] == "all":
            print(f"{row['method']}: MAE={row['mae_m']:.3f} m, RMSE={row['rmse_m']:.3f} m, P90={row['p90_absolute_error_m']:.3f} m")
    print(f"Latency own estimates {[(n, f['own_latency_s']) for n, f in flights.items()]}; applied {latency_used}")
    print(f"Selected {len(records)} records; excluded {dict(excluded)}")


if __name__ == "__main__":
    main()
