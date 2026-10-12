#!/usr/bin/env python3
"""T4.1/T4.2: does the static error mechanism explain the UAV transfer?

Diagnostic only: RTK ground truth stratifies the residuals and, for PDoA, the
oracle mount yaw of evaluate_uav_pdoa.py defines the expected phase.  The
range offset and the PDoA characteristic are frozen from the static campaign.
Criteria are pre-registered in analysis/README.md ("Phase 4").
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from evaluate_uav_range import fix_track, timestamp, track_at
from uav.config import (ANALYSIS, MAX_ALIGNMENT_AGE_S, NODE_ANTENNA_OFFSET_M,
                        TAG_HEIGHT_M, TAG_LAYOUT, UAV_DATA, UAV_PREDICTIONS, UAV_REFERENCE, tag_xy)
from uav.coordinates import q4_heading_deg
from workflow.bootstrap import mean_interval
from workflow.diagnostics import _static_arrays, invert_characteristic
from workflow.phase import wrap
from workflow.phase_elevation import estimated_elevation_deg
from workflow.position_benchmark import write_csv


OUTPUTS = ANALYSIS / "outputs"
OUTPUT = OUTPUTS / "uav_bridge"
STATIC = ANALYSIS / "csv" / "static_measurements.csv"
BLOCK_S = 10.
LAGS_S = np.round(np.arange(-.5, 1. + 1e-9, .05), 2)
HOVER_MPS = .5
SHARED_FACTOR = 2.
LAG_MIN_REDUCTION = .10
ELEVATION_BINS = (0., 10., 25., 45., 90.)
HORIZONTAL_BINS = (0., 10., 20., 30., 45.)
SPEED_BINS = (0., .5, 2., 5., 15.)
AGE_BINS = (0., .025, .05, .075, .1001)
OFF_AXIS_BINS = (0., 15., 35., 55., 90., 180.)


def static_references():
    """Frozen static models and the static test MAE of each model."""
    dev, _, digest = _static_arrays(STATIC, OUTPUTS)
    test, _, _ = _static_arrays(STATIC, OUTPUTS, subsets=("test",))
    frozen = json.loads((OUTPUTS / "final_test" / "manifest.json").read_text())["frozen"]
    characteristic = dict(scale=frozen["physical_scale"], offset_deg=frozen["physical_offset_deg"])
    offset_3d = float(dev["range_error"][dev["quality"]].mean())
    usable = test["quality"]
    range_mae, range_ci, _ = mean_interval(np.abs(test["range_error"][usable] - offset_3d), test["group"][usable])
    elevation_hat = estimated_elevation_deg(test["range_m"], test["height"])
    matched = (np.abs(test["bearing"] - 90.) <= 15.) & (test["range_m"] <= 25.) & (elevation_hat < 45.)
    phase_error = np.abs(wrap(invert_characteristic(test["phase"][matched], characteristic) - test["expected"][matched]))
    phase_mae, phase_ci, _ = mean_interval(phase_error, test["group"][matched])
    return dict(static_source_sha256=digest, range_offset_3d_m=offset_3d, characteristic=characteristic,
                static_test_range_mae_m=range_mae, static_test_range_ci_m=range_ci,
                static_test_matched_pdoa_mae_deg=phase_mae, static_test_matched_pdoa_ci_deg=phase_ci)


def blocks(flight, tag, seconds):
    return np.asarray([f"{f}|{t}|{int(s // BLOCK_S)}" for f, t, s in zip(flight, tag, seconds)])


def strata(name, values, error, groups, edges, unit):
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        select = np.isfinite(values) & (values >= low) & (values < high)
        if select.sum() < 20:
            continue
        mae, (ci_low, ci_high), _ = mean_interval(np.abs(error[select]), groups[select])
        rows.append(dict(variable=name, low=low, high=high, samples=int(select.sum()),
                         **{f"bias_{unit}": float(error[select].mean()), f"mae_{unit}": mae,
                            f"mae_ci_low_{unit}": ci_low, f"mae_ci_high_{unit}": ci_high}))
    return rows


def lag_sweep(offset_3d):
    """3D range residual with the change-point RTK track evaluated at uwb_time + lag.

    Uses the corrected interpolation of evaluate_uav_range.py; lag 0 means no
    latency, so the curve shows how much of the residual timing explains.
    """
    reference = json.loads(UAV_REFERENCE.read_text())
    errors, groups = [], []
    for path in sorted(UAV_DATA.glob("drone_*.csv")):
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        track = fix_track(rows, reference)
        uwb = [row for row in rows if row["rtk_status"] == "50" and float(row["uwb_age_s"]) <= MAX_ALIGNMENT_AGE_S
               and row["tag"] in TAG_LAYOUT]
        when = np.asarray([(timestamp(row["uwb_time"]) - track["origin"]).total_seconds() for row in uwb])
        measured = np.asarray([float(row["range_cm"]) / 100. for row in uwb])
        tags = np.asarray([tag_xy(row["tag"]) for row in uwb])
        matrix = np.full((len(uwb), len(LAGS_S)), np.nan)
        for j, lag in enumerate(LAGS_S):
            x, y, height, _, _, valid = track_at(track, when + lag)
            true_range = np.hypot(np.hypot(x - tags[:, 0], y - tags[:, 1]), height + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M)
            matrix[valid, j] = (measured - true_range - offset_3d)[valid]
        errors.append(matrix)
        groups.append(blocks([path.stem] * len(uwb), [row["tag"] for row in uwb], when))
    errors, groups = np.vstack(errors), np.concatenate(groups)
    paired = np.isfinite(errors).all(axis=1)  # same rows at every lag
    return errors[paired], groups[paired], int((~paired).sum())


def main() -> None:
    references = static_references()
    with UAV_PREDICTIONS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if "uwb_time" not in rows[0]:
        raise SystemExit("Rerun evaluate_uav_range.py: predictions need uwb_time")
    column = lambda name: np.asarray([float(row[name]) for row in rows])
    flight, tag = [row["flight"] for row in rows], [row["tag"] for row in rows]
    origin = min(timestamp(row["uwb_time"]) for row in rows)
    seconds = np.asarray([(timestamp(row["uwb_time"]) - origin).total_seconds() for row in rows])
    groups = blocks(flight, tag, seconds)
    horizontal, true_range = column("horizontal_truth_m"), column("true_range_m")
    elevation = np.rad2deg(np.arctan2(np.sqrt(np.maximum(true_range ** 2 - horizontal ** 2, 0.)), horizontal))
    speed, age = column("speed_mps"), column("uwb_age_s")

    # T4.1 range.
    range_error = column("raw_range_error_m") - references["range_offset_3d_m"]
    range_rows = strata("all", np.zeros(len(rows)), range_error, groups, (0., 1.), "m")
    for name, values, edges in (("elevation_deg", elevation, ELEVATION_BINS), ("horizontal_m", horizontal, HORIZONTAL_BINS),
                                ("speed_mps", speed, SPEED_BINS), ("uwb_age_s", age, AGE_BINS)):
        range_rows += strata(name, values, range_error, groups, edges, "m")
    static_like = (speed < HOVER_MPS) & (elevation < 10.) & (horizontal <= 30.)
    like_mae, like_ci, _ = mean_interval(np.abs(range_error[static_like]), groups[static_like])
    range_verdict = dict(static_like_samples=int(static_like.sum()), uav_static_like_mae_m=like_mae,
                         uav_static_like_ci_m=like_ci, static_test_mae_m=references["static_test_range_mae_m"],
                         shared=bool(like_mae <= SHARED_FACTOR * references["static_test_range_mae_m"]))

    # T4.1 PDoA with the oracle mount.
    oracle = json.loads((OUTPUTS / "uav_pdoa_diagnostic.json").read_text())
    q4 = q4_heading_deg(json.loads(UAV_REFERENCE.read_text()))
    tag_positions = np.asarray([tag_xy(t) for t in tag])
    dx, dy = tag_positions[:, 0] - column("node_x_m"), tag_positions[:, 1] - column("node_y_m")
    theta = wrap(q4 + np.rad2deg(np.arctan2(dx, dy)) - column("yaw_deg") - oracle["best_node_to_aircraft_yaw_offset_deg"])
    expected = oracle["antenna_order_sign"] * 180. * np.sin(np.deg2rad(theta)) * np.hypot(dx, dy) / true_range
    phase_error = wrap(invert_characteristic(column("pdoa_deg"), references["characteristic"]) - expected)
    off_axis = np.abs(theta)  # 0 = front boresight, 180 = behind; static covered only the front half-plane
    pdoa_rows = strata("all", np.zeros(len(rows)), phase_error, groups, (0., 1.), "deg")
    for name, values, edges in (("off_axis_deg", off_axis, OFF_AXIS_BINS), ("horizontal_m", horizontal, HORIZONTAL_BINS),
                                ("elevation_deg", elevation, ELEVATION_BINS), ("speed_mps", speed, SPEED_BINS)):
        pdoa_rows += strata(name, values, phase_error, groups, edges, "deg")
    matched = (off_axis <= 15.) & (horizontal <= 25.) & (elevation < 45.) & (speed < HOVER_MPS)
    pdoa_verdict = dict(matched_samples=int(matched.sum()))
    if matched.sum() >= 20:
        matched_mae, matched_ci, _ = mean_interval(np.abs(phase_error[matched]), groups[matched])
        pdoa_verdict.update(uav_matched_mae_deg=matched_mae, uav_matched_ci_deg=matched_ci,
                            static_test_matched_mae_deg=references["static_test_matched_pdoa_mae_deg"],
                            shared=bool(matched_mae <= SHARED_FACTOR * references["static_test_matched_pdoa_mae_deg"]))

    # T4.2 synchronization lag.
    lag_errors, lag_groups, unpaired = lag_sweep(references["range_offset_3d_m"])
    lag_mae = np.abs(lag_errors).mean(axis=0)
    zero, best = int(np.flatnonzero(LAGS_S == 0.)[0]), int(np.argmin(lag_mae))
    diff, (diff_low, diff_high), _ = mean_interval(np.abs(lag_errors[:, best]) - np.abs(lag_errors[:, zero]), lag_groups)
    lag_rows = [dict(lag_s=float(lag), samples=len(lag_errors), bias_m=float(lag_errors[:, j].mean()),
                     mae_m=float(lag_mae[j])) for j, lag in enumerate(LAGS_S)]
    lag_verdict = dict(best_lag_s=float(LAGS_S[best]), mae_at_zero_m=float(lag_mae[zero]), mae_at_best_m=float(lag_mae[best]),
                       reduction=float(1 - lag_mae[best] / lag_mae[zero]), paired_difference_m=diff,
                       paired_ci_m=(diff_low, diff_high), rows_dropped_unpaired=unpaired)
    lag_verdict["confounder"] = bool(lag_verdict["reduction"] >= LAG_MIN_REDUCTION and diff_high < 0)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "range_strata.csv", range_rows)
    write_csv(OUTPUT / "pdoa_strata.csv", pdoa_rows)
    write_csv(OUTPUT / "lag_sweep.csv", lag_rows)
    manifest = dict(status="complete", test="T4.1/T4.2", kind="diagnostic (RTK ground truth, oracle mount for PDoA)",
                    static_references=references, oracle_mount=oracle, block_seconds=BLOCK_S,
                    range_verdict=range_verdict, pdoa_verdict=pdoa_verdict, lag_verdict=lag_verdict)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(json.dumps(dict(range=range_verdict, pdoa=pdoa_verdict, lag=lag_verdict), indent=2, default=float))


if __name__ == "__main__":
    main()
