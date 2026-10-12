#!/usr/bin/env python3
"""R1/R2: are the UAV range residuals explained by the height reference or the taped tag layout?

Diagnostic (uses RTK ground truth); pre-registered in analysis/README.md.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from evaluate_uav_bridge import blocks, static_references
from evaluate_uav_range import timestamp
from uav.config import ANALYSIS, NODE_ANTENNA_OFFSET_M, TAG_HEIGHT_M, UAV_DATA, UAV_PREDICTIONS, UAV_REFERENCE
from uav.coordinates import q4_heading_deg
from workflow.bootstrap import mean_interval
from workflow.position_benchmark import write_csv


OUTPUT = ANALYSIS / "outputs" / "uav_survey"
GROUND_FUSION_MAX_M = .3
HEIGHT_CONFOUNDER_SHRINK = .5
SURVEY_MIN_REDUCTION = .30


def ground_reference() -> float:
    """Median rtk_h - height_fusion on fixed on-ground rows of all flights."""
    differences = []
    for path in sorted(UAV_DATA.glob("drone_*.csv")):
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                if row["rtk_status"] == "50" and float(row["height_fusion_m"]) < GROUND_FUSION_MAX_M:
                    differences.append(float(row["rtk_h_m"]) - float(row["height_fusion_m"]))
    if not differences:
        raise ValueError("No RTK-fixed on-ground rows to anchor the height reference")
    return float(np.median(differences)), len(differences)


def fit_tag(node_x, node_y, height, measured, start, iterations=20):
    """Gauss-Newton fit of a tag's (x, y) to measured 3D ranges."""
    position = np.asarray(start, dtype=float)
    for _ in range(iterations):
        dx, dy = position[0] - node_x, position[1] - node_y
        predicted = np.sqrt(dx ** 2 + dy ** 2 + height ** 2)
        jacobian = np.column_stack((dx, dy)) / predicted[:, None]
        step, *_ = np.linalg.lstsq(jacobian, measured - predicted, rcond=None)
        position += step
        if np.hypot(*step) < 1e-4:
            break
    return position


def lever_design(yaw_local_deg):
    """Local x/y displacement per unit (forward, right) lever arm at a local heading."""
    heading = np.deg2rad(yaw_local_deg)  # clockwise from local +y
    forward = np.column_stack((np.sin(heading), np.cos(heading)))
    right = np.column_stack((np.cos(heading), -np.sin(heading)))
    return forward, right


def apply_lever_translation(theta, node_x, node_y, yaw_local, tag_x, tag_y, dz):
    """Predicted 3D range for theta = [forward, right, layout dx, layout dy]."""
    forward, right = lever_design(yaw_local)
    nx = node_x + theta[0] * forward[:, 0] + theta[1] * right[:, 0]
    ny = node_y + theta[0] * forward[:, 1] + theta[1] * right[:, 1]
    return np.sqrt((tag_x + theta[2] - nx) ** 2 + (tag_y + theta[3] - ny) ** 2 + dz ** 2)


def fit_lever_translation(node_x, node_y, yaw_local, tag_x, tag_y, dz, measured, iterations=20):
    """Gauss-Newton fit of [forward, right, layout dx, layout dy]."""
    forward, right = lever_design(yaw_local)
    theta = np.zeros(4)
    for _ in range(iterations):
        nx = node_x + theta[0] * forward[:, 0] + theta[1] * right[:, 0]
        ny = node_y + theta[0] * forward[:, 1] + theta[1] * right[:, 1]
        ex, ey = tag_x + theta[2] - nx, tag_y + theta[3] - ny
        predicted = np.sqrt(ex ** 2 + ey ** 2 + dz ** 2)
        ux, uy = ex / predicted, ey / predicted
        jacobian = np.column_stack((-(ux * forward[:, 0] + uy * forward[:, 1]),
                                    -(ux * right[:, 0] + uy * right[:, 1]), ux, uy))
        step, *_ = np.linalg.lstsq(jacobian, measured - predicted, rcond=None)
        theta += step
        if np.abs(step).max() < 1e-5:
            break
    return theta


def main() -> None:
    offset = static_references()["range_offset_3d_m"]
    with UAV_PREDICTIONS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    column = lambda name: np.asarray([float(row[name]) for row in rows])
    flight = np.asarray([row["flight"] for row in rows])
    tags = np.asarray([row["tag"] for row in rows])
    origin = min(timestamp(row["uwb_time"]) for row in rows)
    groups = blocks(flight, tags, [(timestamp(row["uwb_time"]) - origin).total_seconds() for row in rows])
    node_x, node_y, tag_x, tag_y = column("node_x_m"), column("node_y_m"), column("tag_x_m"), column("tag_y_m")
    measured = column("raw_range_m") - offset
    horizontal = np.hypot(tag_x - node_x, tag_y - node_y)

    # R1: fused versus RTK-derived height.
    ground, ground_rows = ground_reference()
    heights = {"height_fusion": column("node_height_m"),
               "rtk_height": column("rtk_ellipsoidal_h_m") - ground}
    r1 = []
    for name, node_height in heights.items():
        error = measured - np.hypot(horizontal, node_height + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M)
        mae, ci, _ = mean_interval(np.abs(error), groups)
        per_flight = {f: float(error[flight == f].mean()) for f in sorted(set(flight))}
        r1.append(dict(height_reference=name, mae_m=mae, mae_ci_low_m=ci[0], mae_ci_high_m=ci[1],
                       **{f"bias_{f[-6:]}_m": b for f, b in per_flight.items()},
                       inter_flight_bias_difference_m=abs(np.subtract(*per_flight.values()))))
    shrink = 1 - r1[1]["inter_flight_bias_difference_m"] / r1[0]["inter_flight_bias_difference_m"]
    r1_verdict = dict(ground_ellipsoidal_h_m=ground, ground_rows=ground_rows, inter_flight_bias_shrink=shrink,
                      height_confounder=bool(shrink >= HEIGHT_CONFOUNDER_SHRINK))

    # R2: tag positions estimated on one flight, evaluated on the other (fused height, as in the pipeline).
    dz = column("node_height_m") + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
    estimates = {}
    for train_flight in sorted(set(flight)):
        for tag in sorted(set(tags)):
            select = (flight == train_flight) & (tags == tag)
            if select.sum() >= 50:
                start = (tag_x[select][0], tag_y[select][0])
                estimates[(train_flight, tag)] = fit_tag(node_x[select], node_y[select], dz[select],
                                                         measured[select], start)
    taped_error = measured - np.hypot(horizontal, dz)
    r2, per_tag = [], []
    for train_flight in sorted(set(flight)):
        test = flight != train_flight
        usable = test & np.asarray([(train_flight, t) in estimates for t in tags])
        surveyed = np.asarray([estimates.get((train_flight, t), (np.nan, np.nan)) for t in tags])
        fitted_error = measured - np.hypot(np.hypot(surveyed[:, 0] - node_x, surveyed[:, 1] - node_y), dz)
        taped_mae, taped_ci, _ = mean_interval(np.abs(taped_error[usable]), groups[usable])
        fitted_mae, fitted_ci, _ = mean_interval(np.abs(fitted_error[usable]), groups[usable])
        diff, diff_ci, _ = mean_interval(np.abs(fitted_error[usable]) - np.abs(taped_error[usable]), groups[usable])
        r2.append(dict(train_flight=train_flight, samples=int(usable.sum()),
                       taped_mae_m=taped_mae, taped_ci_low_m=taped_ci[0], taped_ci_high_m=taped_ci[1],
                       fitted_mae_m=fitted_mae, fitted_ci_low_m=fitted_ci[0], fitted_ci_high_m=fitted_ci[1],
                       paired_difference_m=diff, paired_ci_low_m=diff_ci[0], paired_ci_high_m=diff_ci[1],
                       reduction=1 - fitted_mae / taped_mae))
    flights = sorted(set(flight))
    for tag in sorted(set(tags)):
        taped = np.asarray([tag_x[tags == tag][0], tag_y[tags == tag][0]])
        row = dict(tag=tag, taped_x_m=taped[0], taped_y_m=taped[1])
        corrections = []
        for f in flights:
            if (f, tag) in estimates:
                fitted = estimates[(f, tag)]
                row.update({f"fit_{f[-6:]}_x_m": fitted[0], f"fit_{f[-6:]}_y_m": fitted[1],
                            f"correction_{f[-6:]}_m": float(np.hypot(*(fitted - taped)))})
                corrections.append(fitted)
        row["between_flight_disagreement_m"] = (float(np.hypot(*(corrections[0] - corrections[1])))
                                                if len(corrections) == 2 else None)
        per_tag.append(row)
    magnitudes = [r[k] for r in per_tag for k in r if k.startswith("correction_")]
    disagreements = [r["between_flight_disagreement_m"] for r in per_tag if r["between_flight_disagreement_m"] is not None]
    median_correction, median_disagreement = float(np.median(magnitudes)), float(np.median(disagreements))
    r2_verdict = dict(median_correction_m=median_correction, median_between_flight_disagreement_m=median_disagreement,
                      reductions={r["train_flight"]: r["reduction"] for r in r2},
                      survey_error_supported=bool(all(r["reduction"] >= SURVEY_MIN_REDUCTION and r["paired_ci_high_m"] < 0
                                                      for r in r2) and median_disagreement <= .5 * median_correction))

    # R3: body lever arm + common layout translation, cross-flight.
    yaw_local = column("yaw_deg") - q4_heading_deg(json.loads(UAV_REFERENCE.read_text()))
    r3, thetas = [], {}
    for train_flight in flights:
        train, test = flight == train_flight, flight != train_flight
        theta = fit_lever_translation(node_x[train], node_y[train], yaw_local[train], tag_x[train], tag_y[train],
                                      dz[train], measured[train])
        thetas[train_flight] = theta
        predicted = apply_lever_translation(theta, node_x[test], node_y[test], yaw_local[test], tag_x[test],
                                            tag_y[test], dz[test])
        fitted_error = measured[test] - predicted
        taped_mae, _, _ = mean_interval(np.abs(taped_error[test]), groups[test])
        fitted_mae, fitted_ci, _ = mean_interval(np.abs(fitted_error), groups[test])
        diff, diff_ci, _ = mean_interval(np.abs(fitted_error) - np.abs(taped_error[test]), groups[test])
        r3.append(dict(train_flight=train_flight, forward_m=theta[0], right_m=theta[1], layout_dx_m=theta[2],
                       layout_dy_m=theta[3], taped_mae_m=taped_mae, fitted_mae_m=fitted_mae,
                       fitted_ci_low_m=fitted_ci[0], fitted_ci_high_m=fitted_ci[1], paired_difference_m=diff,
                       paired_ci_low_m=diff_ci[0], paired_ci_high_m=diff_ci[1],
                       bias_after_m=float(fitted_error.mean()), reduction=1 - fitted_mae / taped_mae))
    a, b = thetas.values()
    r3_verdict = dict(parameter_disagreement_m=float(np.linalg.norm(a - b)),
                      median_parameter_magnitude_m=float(np.median([np.linalg.norm(a), np.linalg.norm(b)])))
    r3_verdict["supported"] = bool(
        all(r["reduction"] >= SURVEY_MIN_REDUCTION and r["paired_ci_high_m"] < 0 for r in r3)
        and r3_verdict["parameter_disagreement_m"] <= .5 * r3_verdict["median_parameter_magnitude_m"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "lever_translation_cross_flight.csv", r3)
    write_csv(OUTPUT / "height_reference.csv", r1)
    write_csv(OUTPUT / "tag_survey_cross_flight.csv", r2)
    write_csv(OUTPUT / "tag_positions.csv", per_tag)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", kind="diagnostic (RTK ground truth)", range_offset_3d_m=offset,
        r1=r1_verdict, r2=r2_verdict, r3=r3_verdict), indent=2, default=float) + "\n")
    print(json.dumps(dict(r1_verdict=r1_verdict, r2_verdict=r2_verdict, r3=r3, r3_verdict=r3_verdict),
                     indent=2, default=float))


if __name__ == "__main__":
    main()
