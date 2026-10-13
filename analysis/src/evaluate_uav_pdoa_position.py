#!/usr/bin/env python3
"""UAV bearing and position from PDoA with the frozen static characteristic.

The node is on the aircraft and the tags are fixed.  The mount (yaw offset and
antenna-order sign) is calibrated on one flight and evaluated on the other.
Protocol and criterion are pre-registered in analysis/README.md.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from evaluate_uav_bridge import OUTPUTS, blocks, static_references
from evaluate_uav_range import timestamp
from uav.config import ANALYSIS, NODE_ANTENNA_OFFSET_M, TAG_HEIGHT_M, UAV_PREDICTIONS, UAV_REFERENCE
from uav.coordinates import q4_heading_deg
from workflow.bootstrap import mean_interval
from workflow.diagnostics import invert_characteristic
from workflow.phase import wrap
from workflow.phase_elevation import estimated_elevation_deg
from workflow.position_benchmark import write_csv


OUTPUT = ANALYSIS / "outputs" / "uav_pdoa_position"
MOUNT_GRID_DEG = np.arange(-180., 180., .25)
GATE_RANGE_M, GATE_ELEVATION_DEG = 25., 45.
STATIC_GATED_POSITION_MAE_M = 2.699  # T3.2, offset + physical_inverse, gated
TRANSFER_FACTOR = 2.
# (offset, sign) and (offset + 180, -sign) give identical phase, so PDoA alone
# cannot tell front from rear.  Fix the static antenna-order convention
# (eta = -180 x / r, node not flipped); the unconstrained fit is kept as a diagnostic.
STATIC_SIGN = -1.


def fit_mount(phase, heading_rel, factor, signs=(STATIC_SIGN,)):
    """Yaw offset (and sign) minimizing the phase residual MAE on one flight."""
    best = None
    for sign in signs:
        for offset in MOUNT_GRID_DEG:
            expected = sign * 180. * np.sin(np.deg2rad(heading_rel - offset)) * factor
            score = float(np.abs(wrap(phase - expected)).mean())
            if best is None or score < best[0]:
                best = (score, float(offset), sign)
    return dict(train_phase_mae_deg=best[0], yaw_offset_deg=best[1], sign=best[2])


def main() -> None:
    references = static_references()
    with UAV_PREDICTIONS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    column = lambda name: np.asarray([float(row[name]) for row in rows])
    flight = np.asarray([row["flight"] for row in rows])
    tags = [row["tag"] for row in rows]
    origin = min(timestamp(row["uwb_time"]) for row in rows)
    groups = blocks(flight, tags, [(timestamp(row["uwb_time"]) - origin).total_seconds() for row in rows])
    q4 = q4_heading_deg(json.loads(UAV_REFERENCE.read_text()))

    node_x, node_y, yaw = column("node_x_m"), column("node_y_m"), column("yaw_deg")
    tag_x, tag_y = column("tag_x_m"), column("tag_y_m")
    height = column("node_height_m") + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
    # Truth: local heading of the tag seen from the node (from +y towards +x).
    true_heading = np.rad2deg(np.arctan2(tag_x - node_x, tag_y - node_y))
    true_factor = np.hypot(tag_x - node_x, tag_y - node_y) / column("true_range_m")
    heading_rel_truth = wrap(q4 + true_heading - yaw)
    # Observables: offset-corrected 3D range, fused height, PDoA, yaw.
    r3d = column("raw_range_m") - references["range_offset_3d_m"]
    rho = np.sqrt(np.maximum(r3d ** 2 - height ** 2, 0.))
    factor_hat = np.where(rho > 0, r3d / np.where(rho > 0, rho, 1.), np.inf)
    elevation_hat = estimated_elevation_deg(column("raw_range_m"), height)
    gated = (column("raw_range_m") <= GATE_RANGE_M) & (elevation_hat <= GATE_ELEVATION_DEG)
    phases = {"pdoa_physical": invert_characteristic(column("pdoa_deg"), references["characteristic"]),
              "pdoa_raw": wrap(column("pdoa_deg"))}

    reports, records, mounts = [], [], {}
    for method, phase in phases.items():
        for train_flight in sorted(set(flight)):
            train, test = flight == train_flight, flight != train_flight
            mount = fit_mount(phase[train], heading_rel_truth[train], true_factor[train])
            mounts[f"{method}|{train_flight}"] = dict(
                **mount, unconstrained=fit_mount(phase[train], heading_rel_truth[train], true_factor[train], (-1., 1.)))
            sine = np.clip(mount["sign"] * phase * factor_hat / 180., -1., 1.)
            heading_rel_hat = np.rad2deg(np.arcsin(sine)) + mount["yaw_offset_deg"]  # front hemisphere
            heading_hat = heading_rel_hat - q4 + yaw
            x_hat = node_x + rho * np.sin(np.deg2rad(heading_hat))
            y_hat = node_y + rho * np.cos(np.deg2rad(heading_hat))
            bearing_error = np.abs(wrap(heading_hat - true_heading))
            position_error = np.hypot(x_hat - tag_x, y_hat - tag_y)
            rear = np.abs(wrap(heading_rel_truth - mount["yaw_offset_deg"])) > 90.
            for scope, mask in (("all", test), ("gated", test & gated), ("gated_front", test & gated & ~rear)):
                bearing, bearing_ci, _ = mean_interval(bearing_error[mask], groups[mask])
                position, position_ci, _ = mean_interval(position_error[mask], groups[mask])
                reports.append(dict(method=method, train_flight=train_flight, scope=scope, samples=int(mask.sum()),
                                    coverage=float(mask.sum() / test.sum()), rear_share=float(rear[mask].mean()),
                                    bearing_mae_deg=bearing, bearing_ci_low_deg=bearing_ci[0], bearing_ci_high_deg=bearing_ci[1],
                                    bearing_p90_deg=float(np.quantile(bearing_error[mask], .9, method="inverted_cdf")),
                                    position_mae_m=position, position_ci_low_m=position_ci[0], position_ci_high_m=position_ci[1],
                                    position_p90_m=float(np.quantile(position_error[mask], .9, method="inverted_cdf"))))
            if method == "pdoa_physical":
                records += [dict(train_flight=train_flight, flight=flight[i], tag=tags[i], uwb_time=rows[i]["uwb_time"],
                                 gated=bool(gated[i]), rear=bool(rear[i]), pdoa_deg=float(rows[i]["pdoa_deg"]),
                                 raw_range_m=float(rows[i]["raw_range_m"]), elevation_hat_deg=float(elevation_hat[i]),
                                 tag_x_hat_m=float(x_hat[i]), tag_y_hat_m=float(y_hat[i]),
                                 tag_x_m=float(tag_x[i]), tag_y_m=float(tag_y[i]),
                                 bearing_error_deg=float(bearing_error[i]),
                                 position_error_m=float(position_error[i])) for i in np.flatnonzero(test)]
    primary = [row for row in reports if row["method"] == "pdoa_physical" and row["scope"] == "gated"]
    verdict = dict(static_gated_position_mae_m=STATIC_GATED_POSITION_MAE_M,
                   uav_gated_position_mae_m={row["train_flight"]: row["position_mae_m"] for row in primary},
                   transfer=bool(all(row["position_mae_m"] <= TRANSFER_FACTOR * STATIC_GATED_POSITION_MAE_M
                                     for row in primary)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "metrics.csv", reports)
    write_csv(OUTPUT / "predictions.csv", records)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", protocol="mount fitted on one flight, evaluated on the other",
        static_references=references, mounts=mounts, gate=dict(range_m=GATE_RANGE_M, elevation_deg=GATE_ELEVATION_DEG),
        hemisphere="front assumed; rear share reported", verdict=verdict), indent=2, default=float) + "\n")
    for row in reports:
        print(f"{row['method']} train={row['train_flight'][-6:]} {row['scope']}: n={row['samples']} cov={row['coverage']:.2f} "
              f"rear={row['rear_share']:.2f} bearing={row['bearing_mae_deg']:.1f} deg, "
              f"position={row['position_mae_m']:.2f} m [{row['position_ci_low_m']:.2f}, {row['position_ci_high_m']:.2f}], "
              f"P90={row['position_p90_m']:.2f} m")
    print(json.dumps(dict(mounts=mounts, verdict=verdict), indent=2, default=float))


if __name__ == "__main__":
    main()
