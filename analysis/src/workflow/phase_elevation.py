"""Phase 2: generic PDoA correction with observable elevation and range, gating, position.

Pre-registered before evaluation (see analysis/README.md, "Phase 2"):

* elevation_range model: target cos/sin of the circular residual, as in the
  trigonometric model; design = [1, sin eta, cos eta, sin 2eta, cos 2eta]
  (x) [1, e, e^2, r_std], 20 columns.  e = atan(h / rho_hat) from the measured
  range and the known height; r_std is the measured range standardized on the
  training fold.  No tag, bearing, rotation or ground truth is an input.
* success: MAE at elevation >= 25 deg falls by >= 30% against
  trigonometric_global with a paired interval excluding 0, and below 25 deg
  the paired interval does not lie entirely above 0 (no significant worsening).
* gate: fixed grid over |measured PDoA|, measured range and estimated
  elevation; operating points are the lowest-MAE cells reaching 90/75/50%
  coverage of out-of-fold LODO predictions.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .bootstrap import configuration_keys, mean_interval
from .diagnostics import _static_arrays, fit_phase_characteristic, invert_characteristic
from .phase import circular_summary, fit_phase_offset, wrap
from .phase_trigonometric import design as harmonic_basis, fit_trigonometric, predict_correction
from .position_benchmark import metrics as position_metrics, reconstruct, write_csv


METHODS = ("raw", "trigonometric_global", "physical_inverse", "elevation_range")
SUCCESS_ELEVATION_DEG = 25.
SUCCESS_REDUCTION = .30
GATE_PHASE_DEG = (180., 150., 120., 90., 60., 45., 30.)
GATE_RANGE_M = (31., 25., 20., 15., 10.)
GATE_ELEVATION_DEG = (90., 45., 30.)
COVERAGE_TARGETS = (.90, .75, .50)
OPERATING_COVERAGE = .75
PROTOCOL_KEYS = {"lodo": "distance", "loho": "height", "lobo": "tag"}


def estimated_elevation_deg(range_m, height):
    """Elevation implied by measured 3D range and known height (90 deg if range < height)."""
    ground = np.sqrt(np.maximum(np.asarray(range_m) ** 2 - np.asarray(height) ** 2, 0.))
    return np.rad2deg(np.arctan2(height, ground))


def _design(phase, elevation_deg, range_m, center, scale):
    e = np.deg2rad(elevation_deg)
    modifiers = np.column_stack((np.ones_like(e), e, e ** 2, (range_m - center) / scale))
    base = harmonic_basis(phase)
    return (base[:, :, None] * modifiers[:, None, :]).reshape(len(base), -1)


def fit_elevation_range(phase, expected, elevation_deg, range_m):
    center, scale = float(range_m.mean()), float(range_m.std())
    X = _design(phase, elevation_deg, range_m, center, scale)
    residual = np.deg2rad(wrap(phase - expected))
    coefficients, _, rank, _ = np.linalg.lstsq(X, np.column_stack((np.cos(residual), np.sin(residual))), rcond=None)
    if rank != X.shape[1]:
        raise ValueError("Elevation/range design is rank deficient on this training fold")
    return dict(center=center, scale=scale, coefficients=coefficients,
                fallback_offset_deg=fit_phase_offset(phase, expected, np.full(len(phase), "train")))


def predict_elevation_range(phase, elevation_deg, range_m, model):
    vectors = _design(phase, elevation_deg, range_m, model["center"], model["scale"]) @ model["coefficients"]
    correction = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    fallback = np.linalg.norm(vectors, axis=1) < 1e-8
    correction[fallback] = model["fallback_offset_deg"]
    return wrap(phase - correction), int(fallback.sum())


def _start(destination: Path, **manifest) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "manifest.json"
    path.write_text(json.dumps(dict(status="running", **manifest), indent=2, default=str) + "\n")
    return path


def _finish(path: Path, **extra) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(status="complete", **extra)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def _paired(a, b, groups):
    """Mean of a - b with grouped interval (same replicates for both)."""
    estimate, (low, high), _ = mean_interval(a - b, groups)
    return estimate, low, high


def run_phase_elevation(dataset: Path, output: Path, protocol: str) -> list[dict]:
    """T2.2 (and the PDoA half of T2.1 for protocol='loho')."""
    data, split, digest = _static_arrays(dataset, output)
    key = data[PROTOCOL_KEYS[protocol]]
    elevation_hat = estimated_elevation_deg(data["range_m"], data["height"])
    manifest_path = _start(output / f"pdoa_elevation_{protocol}", source_sha256=digest, split=split,
                           protocol=protocol, held_out=PROTOCOL_KEYS[protocol], methods=list(METHODS),
                           inputs=["pdoa_deg", "range_m", "height_difference_m"],
                           forbidden_inputs=["tag", "bearing", "tag_rotation_deg", "ground-truth geometry"],
                           pre_registered=__doc__, development_pool="train + validation; grouped test excluded",
                           folds=[])
    phase, expected = data["phase"], data["expected"]
    estimates = {method: np.empty(len(phase)) for method in METHODS}
    for held in sorted(set(key)):
        train, evaluate = key != held, key == held
        everything_train = np.full(train.sum(), "train", dtype=object)
        trig = fit_trigonometric(phase[train], expected[train], everything_train)
        correction, _, _ = predict_correction(phase[evaluate], trig)
        physical = fit_phase_characteristic(phase[train], expected[train], data["group"][train], n_boot=0)
        model = fit_elevation_range(phase[train], expected[train], elevation_hat[train], data["range_m"][train])
        estimates["raw"][evaluate] = wrap(phase[evaluate])
        estimates["trigonometric_global"][evaluate] = wrap(phase[evaluate] - correction)
        estimates["physical_inverse"][evaluate] = invert_characteristic(phase[evaluate], physical)
        estimates["elevation_range"][evaluate], fallback = predict_elevation_range(
            phase[evaluate], elevation_hat[evaluate], data["range_m"][evaluate], model)
        manifest = json.loads(manifest_path.read_text())
        manifest["folds"].append(dict(held=str(held), training_samples=int(train.sum()),
                                      evaluated_samples=int(evaluate.sum()), elevation_range_fallbacks=fallback,
                                      physical_scale=physical["scale"], physical_offset_deg=physical["offset_deg"]))
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(f"PDoA {protocol} {held} complete", flush=True)
    errors = {method: wrap(estimate - expected) for method, estimate in estimates.items()}
    groups = data["group"]
    high = data["elevation"] >= SUCCESS_ELEVATION_DEG
    summary = []
    for stratum, select in (("all", np.ones(len(phase), dtype=bool)),
                            (f"elevation<{SUCCESS_ELEVATION_DEG:g}", ~high),
                            (f"elevation>={SUCCESS_ELEVATION_DEG:g}", high)):
        if not select.any():
            continue
        baseline = np.abs(errors["trigonometric_global"][select])
        for method, error in errors.items():
            absolute = np.abs(error[select])
            mae, (low, high_ci), _ = mean_interval(absolute, groups[select])
            diff, diff_low, diff_high = _paired(absolute, baseline, groups[select])
            summary.append(dict(stratum=stratum, method=method, **circular_summary(error[select]),
                                mae_ci_low_deg=low, mae_ci_high_deg=high_ci,
                                minus_trigonometric_global_mae_deg=diff,
                                minus_trigonometric_global_ci_low_deg=diff_low,
                                minus_trigonometric_global_ci_high_deg=diff_high))
    by_fold = [dict(held=str(held), method=method, **circular_summary(error[key == held]))
               for held in sorted(set(key)) for method, error in errors.items()]
    predictions = [dict(csv_line=int(data["csv_line"][i]), tag=data["tag"][i], height_difference_m=data["height"][i],
                        target_ground_distance_m=data["distance"][i], tag_rotation_deg=data["rotation"][i],
                        gt_x_m=data["gt_x"][i], gt_y_m=data["gt_y"][i],
                        range_m=data["range_m"][i], estimated_elevation_deg=elevation_hat[i],
                        expected_phase_deg=expected[i], **{f"{m}_phase_deg": estimates[m][i] for m in METHODS})
                   for i in range(len(phase))]
    destination = manifest_path.parent
    write_csv(destination / "predictions.csv", predictions)
    write_csv(destination / "metrics.csv", summary)
    write_csv(destination / "metrics_by_fold.csv", by_fold)
    verdict = None
    rows = {(row["stratum"], row["method"]): row for row in summary}
    hi, lo = f"elevation>={SUCCESS_ELEVATION_DEG:g}", f"elevation<{SUCCESS_ELEVATION_DEG:g}"
    if (hi, "elevation_range") in rows:
        high_row, base_row, low_row = rows[(hi, "elevation_range")], rows[(hi, "trigonometric_global")], rows[(lo, "elevation_range")]
        verdict = dict(high_elevation_reduction=1 - high_row["mae_deg"] / base_row["mae_deg"],
                       high_elevation_paired_interval_excludes_zero=high_row["minus_trigonometric_global_ci_high_deg"] < 0,
                       low_elevation_not_worse=low_row["minus_trigonometric_global_ci_low_deg"] <= 0)
        verdict["success"] = bool(verdict["high_elevation_reduction"] >= SUCCESS_REDUCTION
                                  and verdict["high_elevation_paired_interval_excludes_zero"]
                                  and verdict["low_elevation_not_worse"])
    _finish(manifest_path, verdict=verdict)
    return summary


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def gate_mask(phase_deg, range_m, elevation_deg, cell):
    phase_limit, range_limit, elevation_limit = cell
    return (np.abs(wrap(phase_deg)) <= phase_limit) & (range_m <= range_limit) & (elevation_deg <= elevation_limit)


def run_gate(output: Path) -> list[dict]:
    """T2.3: coverage-error trade-off of an observable-only reliability gate (LODO, out-of-fold)."""
    source = output / "pdoa_elevation_lodo" / "predictions.csv"
    if not source.exists():
        raise FileNotFoundError("Run --pdoa-elevation-lodo before the gate")
    rows = _read(source)
    column = lambda name: np.asarray([float(row[name]) for row in rows])
    phase, range_m, elevation, expected = (column(name) for name in
                                           ("raw_phase_deg", "range_m", "estimated_elevation_deg", "expected_phase_deg"))
    groups = configuration_keys([row["tag"] for row in rows], column("height_difference_m"),
                                column("target_ground_distance_m"), column("tag_rotation_deg"))
    errors = {method: np.abs(wrap(column(f"{method}_phase_deg") - expected))
              for method in ("trigonometric_global", "elevation_range")}
    manifest_path = _start(output / "pdoa_gate", source=str(source.relative_to(output)), test="T2.3",
                           observables=["|measured pdoa|", "measured range", "elevation from measured range"],
                           grid=dict(phase_deg=GATE_PHASE_DEG, range_m=GATE_RANGE_M, elevation_deg=GATE_ELEVATION_DEG),
                           selection="lowest MAE cell reaching each coverage target, on LODO out-of-fold rows",
                           caveat="Operating points are selected on the development pool; confirm on the grouped test.")
    curve = []
    for cell in itertools.product(GATE_PHASE_DEG, GATE_RANGE_M, GATE_ELEVATION_DEG):
        keep = gate_mask(phase, range_m, elevation, cell)
        if keep.any():
            curve.append(dict(phase_limit_deg=cell[0], range_limit_m=cell[1], elevation_limit_deg=cell[2],
                              coverage=float(keep.mean()),
                              **{f"{m}_mae_deg": float(e[keep].mean()) for m, e in errors.items()},
                              **{f"{m}_p90_deg": float(np.quantile(e[keep], .9, method="inverted_cdf"))
                                 for m, e in errors.items()}))
    points = []
    for target in COVERAGE_TARGETS:
        best = min((row for row in curve if row["coverage"] >= target), key=lambda row: row["elevation_range_mae_deg"])
        keep = gate_mask(phase, range_m, elevation,
                         (best["phase_limit_deg"], best["range_limit_m"], best["elevation_limit_deg"]))
        mae, (low, high), _ = mean_interval(errors["elevation_range"][keep], groups[keep])
        points.append(dict(coverage_target=target, **best, elevation_range_mae_ci_low_deg=low,
                           elevation_range_mae_ci_high_deg=high))
    write_csv(manifest_path.parent / "curve.csv", curve)
    write_csv(manifest_path.parent / "operating_points.csv", points)
    _finish(manifest_path, operating_points=points)
    return points


def run_position(output: Path, protocol: str) -> list[dict]:
    """T2.4: position from out-of-fold offset range and PDoA, with and without the gate."""
    phase_source = output / f"pdoa_elevation_{protocol}" / "predictions.csv"
    range_paths = sorted((output / f"distance_{protocol}").glob("*/predictions.csv"))
    gate_manifest = output / "pdoa_gate" / "manifest.json"
    if not phase_source.exists() or not range_paths or not gate_manifest.exists():
        raise FileNotFoundError(f"Run distance/PDoA {protocol} and the gate before position evaluation")
    ranges = {int(row["csv_line"]): row for path in range_paths for row in _read(path)}
    rows = [row for row in _read(phase_source) if int(row["csv_line"]) in ranges]
    point = next(p for p in json.loads(gate_manifest.read_text())["operating_points"]
                 if p["coverage_target"] == OPERATING_COVERAGE)
    cell = (point["phase_limit_deg"], point["range_limit_m"], point["elevation_limit_deg"])
    column = lambda name: np.asarray([float(row[name]) for row in rows])
    height = column("height_difference_m")
    rho = np.asarray([float(ranges[int(row["csv_line"])]["constant_offset_m"]) for row in rows])
    tags = [row["tag"] for row in rows]
    x_true, y_true = column("gt_x_m"), column("gt_y_m")
    groups = configuration_keys(tags, height, column("target_ground_distance_m"), column("tag_rotation_deg"))
    keep = gate_mask(column("raw_phase_deg"), column("range_m"), column("estimated_elevation_deg"), cell)
    manifest_path = _start(output / f"position_elevation_{protocol}", protocol=protocol, test="T2.4",
                           range_prediction=f"constant_offset_m from distance_{protocol}",
                           phase_prediction=f"pdoa_elevation_{protocol}", gate_cell=cell,
                           gate_coverage_target=OPERATING_COVERAGE)
    summary = []
    for method in METHODS:
        x_hat, y_hat, clipped = reconstruct(rho, height, column(f"{method}_phase_deg"))
        error = np.hypot(x_hat - x_true, y_hat - y_true)
        for scope, select in (("all", np.ones(len(rows), dtype=bool)), ("gated", keep)):
            for h in [None, *sorted(set(height))]:
                part = select & (True if h is None else height == h)
                if not part.any():
                    continue
                _, (low, high), _ = mean_interval(error[part], groups[part])
                stratum = np.ones(len(rows), dtype=bool) if h is None else height == h
                summary.append(dict(method=method, scope=scope, height_difference_m="all" if h is None else h,
                                    coverage=float(part.sum() / stratum.sum()),
                                    **position_metrics(x_true[part], y_true[part], x_hat[part], y_hat[part], clipped[part]),
                                    position_mae_ci_low_m=low, position_mae_ci_high_m=high))
    write_csv(manifest_path.parent / "metrics.csv", summary)
    _finish(manifest_path)
    return summary
