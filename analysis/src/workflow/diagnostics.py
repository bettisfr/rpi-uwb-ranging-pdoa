"""Integrity diagnostics (phase 0) and physical PDoA characterization (phase 1).

Phase 0: firmware frame, MLP LOBO failure, bearing-label leakage.
Phase 1: no learned correction; the measured PDoA is described against the
far-field expectation by elevation, rotation and session, with grouped
bootstrap intervals (see bootstrap.py).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .bootstrap import (N_BOOT, SEED, bootstrap_weights, circular_mean_interval, configuration_keys,
                        mean_interval, percentile_interval, wrapped_interval)
from .offset import load_assigned_rows
from .phase import circular_summary, wrap
from .phase_trigonometric import fit_trigonometric, predict_correction
from .phase_continuous import base_features, fit_harmonic, fit_torch_mlp, predict_harmonic
from .position_benchmark import metrics as position_metrics, write_csv


def _column(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def _start(destination: Path, **manifest) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "manifest.json"
    path.write_text(json.dumps(dict(status="running", **manifest), indent=2) + "\n")
    return path


def _finish(path: Path, **extra) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(status="complete", **extra)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def fit_rigid_transform(source, target):
    """Least-squares rotation (reflection allowed) plus translation, no scale."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vt
    return rotation, target_mean - source_mean @ rotation


def run_firmware_frame(dataset: Path, output: Path) -> list[dict]:
    """T0.1: is the firmware position error a frame convention rather than estimation error?"""
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in ("train", "validation")]
    subset = np.asarray([row["subset"] for row in rows])
    train, validation = subset == "train", subset == "validation"
    firmware = np.column_stack((_column(rows, "x_m"), _column(rows, "y_m")))
    truth = np.column_stack((_column(rows, "gt_x_m"), _column(rows, "gt_y_m")))
    height, phase, range_m = (_column(rows, name) for name in ("height_difference_m", "pdoa_deg", "range_m"))
    manifest_path = _start(output / "t0_1_firmware_frame", source_sha256=digest, split=split,
                           test="T0.1", fit_subset="train", evaluated_subset="validation")
    rotation, translation = fit_rigid_transform(firmware[train], truth[train])
    transformed = firmware @ rotation + translation
    # Firmware phase-to-angle map: x/r = clip(sign * scale * wrap(pdoa - offset) / 180).
    # The offset wraps, so a linear fit is meaningless; grid-search it on train.
    radius = np.hypot(firmware[:, 0], firmware[:, 1])
    firmware_sin = firmware[:, 0] / np.maximum(radius, 1e-9)
    best = None
    for sign in (1., -1.):
        for offset in np.arange(-180., 180., .5):
            u = sign * wrap(phase[train] - offset) / 180.
            scale = float(u @ firmware_sin[train] / (u @ u))
            residual = np.median(np.abs(firmware_sin[train] - np.clip(scale * u, -1, 1)))
            if best is None or residual < best[0]:
                best = (residual, sign, offset, scale)
    _, sign, offset, scale = best
    sin_residual = firmware_sin[validation] - np.clip(sign * scale * wrap(phase[validation] - offset) / 180., -1, 1)
    no_clip = np.zeros(len(rows), dtype=bool)
    summary = []
    for name, estimate in (("firmware_raw", firmware), ("firmware_rigid_transform", transformed)):
        for h in [None, *sorted(set(height))]:
            select = validation & (True if h is None else height == h)
            summary.append(dict(method=name, height_difference_m="all" if h is None else h,
                                **position_metrics(truth[select, 0], truth[select, 1],
                                                   estimate[select, 0], estimate[select, 1], no_clip[select])))
    write_csv(manifest_path.parent / "metrics.csv", summary)
    _finish(manifest_path,
            rigid_rotation=rotation.tolist(), rigid_translation_m=translation.tolist(),
            rigid_reflection=bool(np.linalg.det(rotation) < 0),
            rigid_rotation_deg=float(np.rad2deg(np.arctan2(rotation[0, 1], rotation[0, 0]))),
            firmware_sin_vs_pdoa=dict(sign=sign, offset_deg=float(offset), scale=scale,
                                      validation_residual_median=float(np.median(np.abs(sin_residual))),
                                      validation_share_within_0_02=float(np.mean(np.abs(sin_residual) < .02)),
                                      model="x_fw/|xy_fw| = clip(sign * scale * wrap(pdoa_deg - offset_deg) / 180, -1, 1)",
                                      nominal="sign=-1, offset=0, scale=1 for the half-wavelength far-field model"),
            firmware_radius_over_range_median=float(np.median(radius[validation] / range_m[validation])))
    return summary


def run_mlp_check(dataset: Path, output: Path, train_probe=20000) -> list[dict]:
    """T0.2: separate an MLP fitting bug from extrapolation to an unseen bearing."""
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in ("train", "validation")]
    subset = np.asarray([row["subset"] for row in rows])
    tags = np.asarray([row["tag"] for row in rows])
    expected = _column(rows, "eta_expected_deg")
    features = base_features(_column(rows, "pdoa_deg"), _column(rows, "range_m"), _column(rows, "height_difference_m"))
    manifest_path = _start(output / "t0_2_mlp_check", source_sha256=digest, split=split, test="T0.2",
                           features=["sin/cos(pdoa)", "sin/cos(2 pdoa)", "range_m", "height_difference_m"],
                           target="cos/sin(eta_expected_deg)", train_probe_samples=train_probe)
    rng = np.random.default_rng(42)
    folds = [("grouped_validation", subset == "train", subset == "validation")]
    folds += [(f"lobo_{tag}", tags != tag, tags == tag) for tag in sorted(set(tags))]
    results = []
    for fold, train, evaluate in folds:
        probe = rng.choice(np.flatnonzero(train), size=min(train_probe, int(train.sum())), replace=False)
        stacked = np.vstack((features[probe], features[evaluate]))
        harmonic = predict_harmonic(stacked, fit_harmonic(features[train], expected[train]))
        mlp, info = fit_torch_mlp(features[train], expected[train], stacked)
        for method, prediction in (("harmonic_range_height", harmonic), ("mlp_range_height_cuda", mlp)):
            train_error = wrap(prediction[:len(probe)] - expected[probe])
            held_error = wrap(prediction[len(probe):] - expected[evaluate])
            results.append(dict(fold_id=fold, method=method,
                                position=("grouped" if fold == "grouped_validation" else
                                          "extreme" if fold in ("lobo_dw00", "lobo_dw08") else "inner"),
                                train_mae_deg=float(np.abs(train_error).mean()),
                                evaluation_mae_deg=float(np.abs(held_error).mean()),
                                evaluation_p90_deg=float(np.quantile(np.abs(held_error), .9, method="inverted_cdf")),
                                mlp_final_training_loss=info["final_training_loss"] if method.startswith("mlp") else None))
        print(f"T0.2 {fold} complete", flush=True)
        write_csv(manifest_path.parent / "metrics.csv", results)
    _finish(manifest_path)
    return results


def run_bearing_label_check(output: Path) -> list[dict]:
    """T0.3: does the per-bearing correction ignore PDoA and reproduce a per-tag constant?"""
    source = output / "pdoa_stratified_lodo" / "predictions.csv"
    if not source.exists():
        raise FileNotFoundError("Run --lodo-phase-stratified before the bearing-label check")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    tags = np.asarray([row["tag"] for row in rows])
    height = _column(rows, "height_difference_m")
    distance = _column(rows, "held_distance_m")
    expected = _column(rows, "expected_phase_deg")
    # Label-only baseline: per-tag circular mean of the training targets, PDoA unused.
    constant = np.empty(len(rows))
    for held in sorted(set(distance)):
        for tag in sorted(set(tags)):
            train, evaluate = (distance != held) & (tags == tag), (distance == held) & (tags == tag)
            constant[evaluate] = circular_summary(expected[train])["circular_bias_deg"]
    errors = {"label_only_constant": wrap(constant - expected)}
    for method in ("raw", "trigonometric_global", "trigonometric_bearing"):
        errors[method] = wrap(_column(rows, f"{method}_phase_deg") - expected)
    manifest_path = _start(output / "t0_3_bearing_label", test="T0.3", source=str(source.relative_to(output)),
                           protocol="leave_one_distance_out, folds taken from pdoa_stratified_lodo",
                           baseline="per-tag circular mean of training eta_expected_deg; PDoA not used")
    summary = []
    for h in [None, *sorted(set(height))]:
        select = np.ones(len(rows), dtype=bool) if h is None else height == h
        for method, error in errors.items():
            paired = np.abs(errors["trigonometric_bearing"][select]) - np.abs(error[select])
            summary.append(dict(height_difference_m="all" if h is None else h, method=method,
                                **circular_summary(error[select]),
                                trig_bearing_minus_method_mae_deg=float(paired.mean())))
    by_tag = [dict(tag=tag, height_difference_m=h, method=method,
                   mae_deg=float(np.abs(error[(tags == tag) & (height == h)]).mean()))
              for tag in sorted(set(tags)) for h in sorted(set(height)) for method, error in errors.items()]
    write_csv(manifest_path.parent / "metrics.csv", summary)
    write_csv(manifest_path.parent / "metrics_by_tag_height.csv", by_tag)
    _finish(manifest_path)
    return summary


# --- Phase 1: physical PDoA characterization -------------------------------

# Fit window for the far-field characteristic; elevation = atan(h / ground distance).
LOW_ELEVATION_DEG = 15.
# Elevation below which a sample is considered coplanar for rotation/session tests.
COPLANAR_ELEVATION_DEG = 10.
ELEVATION_BINS_DEG = (0., 5., 10., 15., 20., 25., 30., 40., 50., 65.)
SCALE_GRID = np.arange(.5, 1.5 + 1e-9, .0025)
# Distances at which every height is below COPLANAR_ELEVATION_DEG (atan(4/24) = 9.5 deg).
SESSION_MIN_DISTANCE_M = 24.
USEFUL_SECTOR_MAE_DEG = 10.


def _static_arrays(dataset: Path, output: Path, subsets=("train", "validation")):
    """Selected subsets as arrays; by default the development pool, grouped test excluded."""
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in subsets]
    data = dict(subset=np.asarray([row["subset"] for row in rows]),
                tag=np.asarray([row["tag"] for row in rows]),
                height=_column(rows, "height_difference_m"),
                distance=_column(rows, "target_ground_distance_m"),
                rotation=_column(rows, "tag_rotation_deg"),
                bearing=_column(rows, "gt_bearing_deg"),
                phase=_column(rows, "pdoa_deg"),
                expected=_column(rows, "eta_expected_deg"),
                range_m=_column(rows, "range_m"),
                gt_x=_column(rows, "gt_x_m"), gt_y=_column(rows, "gt_y_m"),
                range_error=_column(rows, "range_error_3d_m"),
                csv_line=np.asarray([int(row["csv_line"]) for row in rows]),
                quality=np.asarray([row["quality_enabled"] for row in rows]),
                ground_hat=np.asarray([float(row["estimated_ground_distance_m"] or "nan") for row in rows]),
                firmware_x=_column(rows, "x_m"), firmware_y=_column(rows, "y_m"))
    data["elevation"] = np.rad2deg(np.arctan2(data["height"], data["distance"]))
    data["group"] = configuration_keys(data["tag"], data["height"], data["distance"], data["rotation"])
    return data, split, digest


def fit_phase_characteristic(phase, expected, groups, n_boot=N_BOOT, seed=SEED):
    """Fit measured = wrap(scale * expected + offset) by maximum circular resultant.

    For each scale on a fixed grid the optimal offset is the circular mean of
    wrap(measured - scale * expected).  Per-group resultant vectors make the
    grouped bootstrap a single matrix product.
    """
    _, index = np.unique(groups, return_inverse=True)
    n_groups = index.max() + 1
    sums = np.empty((n_groups, len(SCALE_GRID)), dtype=complex)
    for j, scale in enumerate(SCALE_GRID):
        z = np.exp(1j * np.deg2rad(phase - scale * expected))
        sums[:, j] = np.bincount(index, z.real, n_groups) + 1j * np.bincount(index, z.imag, n_groups)
    total = sums.sum(axis=0)
    best = int(np.argmax(np.abs(total)))
    offset = float(wrap(np.rad2deg(np.angle(total[best]))))
    result = dict(scale=float(SCALE_GRID[best]), offset_deg=offset,
                  resultant_length=float(abs(total[best]) / len(phase)),
                  scale_at_grid_edge=best in (0, len(SCALE_GRID) - 1))
    if n_boot:
        replicate = bootstrap_weights(n_groups, n_boot, seed) @ sums
        chosen = np.argmax(np.abs(replicate), axis=1)
        scale_low, scale_high = percentile_interval(SCALE_GRID[chosen])
        offset_low, offset_high = wrapped_interval(
            offset, np.rad2deg(np.angle(replicate[np.arange(n_boot), chosen])))
        result.update(scale_ci_low=scale_low, scale_ci_high=scale_high,
                      offset_ci_low_deg=offset_low, offset_ci_high_deg=offset_high)
    return result


def invert_characteristic(phase, model):
    """Expected phase implied by the fitted characteristic.

    With scale < 1, |phase| near 180 deg maps beyond +/-180; that is endfire,
    so saturate there instead of wrapping onto the opposite side.
    """
    return np.clip(wrap(phase - model["offset_deg"]) / model["scale"], -180., 180.)


def _error_row(error, groups, **labels):
    mae, (low, high), _ = mean_interval(np.abs(error), groups)
    return dict(**labels, **circular_summary(error), mae_ci_low_deg=low, mae_ci_high_deg=high)


def run_phase_characteristic(dataset: Path, output: Path) -> list[dict]:
    """T1.1: slope and offset of the measured PDoA characteristic at low elevation."""
    data, split, digest = _static_arrays(dataset, output)
    train, validation = data["subset"] == "train", data["subset"] == "validation"
    low = data["elevation"] < LOW_ELEVATION_DEG
    manifest_path = _start(output / "t1_1_phase_characteristic", source_sha256=digest, split=split, test="T1.1",
                           model="pdoa = wrap(scale * eta_expected + offset)", fit_subset="train",
                           fit_window=f"elevation < {LOW_ELEVATION_DEG:g} deg", scale_grid=[.5, 1.5, .0025],
                           bootstrap=dict(replicates=N_BOOT, seed=SEED,
                                          unit="configuration (tag, height, distance, rotation)"))
    fits = []
    for h in [None, *sorted(set(data["height"]))]:
        select = train & low & (True if h is None else data["height"] == h)
        fits.append(dict(height_difference_m="all" if h is None else h, samples=int(select.sum()),
                         **fit_phase_characteristic(data["phase"][select], data["expected"][select],
                                                    data["group"][select])))
        print(f"T1.1 fit h={fits[-1]['height_difference_m']} complete", flush=True)
    model = fits[0]
    fit = train & low
    trig = fit_trigonometric(data["phase"][fit], data["expected"][fit], np.full(fit.sum(), "train", dtype=object))
    metrics = []
    for scope, select in (("low_elevation", validation & low), ("all_elevations", validation)):
        phase, expected, groups = data["phase"][select], data["expected"][select], data["group"][select]
        correction, _, _ = predict_correction(phase, trig)
        for method, estimate in (("raw", wrap(phase)), ("physical_inverse", invert_characteristic(phase, model)),
                                 ("trigonometric_global", wrap(phase - correction))):
            metrics.append(_error_row(wrap(estimate - expected), groups, scope=scope, method=method))
    write_csv(manifest_path.parent / "fits.csv", fits)
    write_csv(manifest_path.parent / "metrics.csv", metrics)
    _finish(manifest_path, pooled_model={k: model[k] for k in ("scale", "offset_deg")})
    return fits + metrics


def _contiguous_sector(table, side):
    """Largest boresight angle reachable from the centre through useful tags only."""
    reach = 0.
    candidates = [row for row in table if np.sign(row["boresight_deg"]) in (0, side)]
    for row in sorted(candidates, key=lambda row: abs(row["boresight_deg"])):
        if not row["useful"]:
            break
        reach = abs(row["boresight_deg"])
    return reach


def run_invertibility(dataset: Path, output: Path) -> list[dict]:
    """T1.2: which bearings can be recovered from PDoA without a tag label."""
    data, split, digest = _static_arrays(dataset, output)
    train, validation = data["subset"] == "train", data["subset"] == "validation"
    low = data["elevation"] < LOW_ELEVATION_DEG
    fit = train & low
    model = fit_phase_characteristic(data["phase"][fit], data["expected"][fit], data["group"][fit], n_boot=0)
    trig = fit_trigonometric(data["phase"][fit], data["expected"][fit], np.full(fit.sum(), "train", dtype=object))
    manifest_path = _start(output / "t1_2_invertibility", source_sha256=digest, split=split, test="T1.2",
                           evaluated="validation, elevation < 15 deg", characteristic=model,
                           useful_definition=f"physical-inverse MAE < {USEFUL_SECTOR_MAE_DEG:g} deg",
                           boresight="tag bearing - 90 deg")
    per_tag = []
    for tag in sorted(set(data["tag"])):
        select = validation & low & (data["tag"] == tag)
        phase, expected, groups = data["phase"][select], data["expected"][select], data["group"][select]
        correction, _, _ = predict_correction(phase, trig)
        physical = wrap(invert_characteristic(phase, model) - expected)
        mae, (low_ci, high_ci), _ = mean_interval(np.abs(physical), groups)
        measured = circular_summary(phase)
        per_tag.append(dict(tag=tag, boresight_deg=float(data["bearing"][select][0] - 90.), samples=int(select.sum()),
                            expected_mean_deg=circular_summary(expected)["circular_bias_deg"],
                            measured_mean_deg=measured["circular_bias_deg"],
                            measured_resultant_length=measured["resultant_length"],
                            raw_mae_deg=float(np.abs(wrap(phase - expected)).mean()),
                            physical_inverse_mae_deg=mae, physical_inverse_ci_low_deg=low_ci,
                            physical_inverse_ci_high_deg=high_ci,
                            trigonometric_global_mae_deg=float(np.abs(wrap(phase - correction - expected)).mean()),
                            useful=bool(mae < USEFUL_SECTOR_MAE_DEG)))
    # Ambiguity: spread of the true expected phase behind each measured-phase bin.
    measured = wrap(data["phase"])
    bins = []
    for start in np.arange(-180., 180., 20.):
        select = low & (measured >= start) & (measured < start + 20.)
        if select.any():
            summary = circular_summary(data["expected"][select])
            bins.append(dict(measured_bin_start_deg=start, samples=int(select.sum()),
                             expected_circular_mean_deg=summary["circular_bias_deg"],
                             expected_circular_std_deg=summary["circular_std_deg"]))
    write_csv(manifest_path.parent / "by_tag.csv", per_tag)
    write_csv(manifest_path.parent / "ambiguity_by_measured_phase.csv", bins)
    _finish(manifest_path, useful_sector_negative_side_deg=_contiguous_sector(per_tag, -1),
            useful_sector_positive_side_deg=_contiguous_sector(per_tag, 1))
    return per_tag


def run_elevation(output: Path) -> list[dict]:
    """T1.3: is the high-height degradation explained by elevation alone?

    Uses the existing out-of-fold LODO predictions.  The matched comparison
    pairs (h=2, d) with (h=4, 2d): identical elevation, double height and
    distance.  h=0 at d and 2d controls for the distance change itself.
    """
    source = output / "pdoa_stratified_lodo" / "predictions.csv"
    if not source.exists():
        raise FileNotFoundError("Run --lodo-phase-stratified before the elevation test")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    tag = np.asarray([row["tag"] for row in rows])
    height, distance = _column(rows, "height_difference_m"), _column(rows, "held_distance_m")
    rotation, expected = _column(rows, "tag_rotation_deg"), _column(rows, "expected_phase_deg")
    groups = configuration_keys(tag, height, distance, rotation)
    elevation = np.rad2deg(np.arctan2(height, distance))
    errors = {method: np.abs(wrap(_column(rows, f"{method}_phase_deg") - expected))
              for method in ("raw", "trigonometric_global")}
    manifest_path = _start(output / "t1_3_elevation", test="T1.3", source=str(source.relative_to(output)),
                           elevation="atan(height / nominal ground distance)", bins_deg=list(ELEVATION_BINS_DEG),
                           bootstrap=dict(replicates=N_BOOT, seed=SEED, unit="configuration"))
    by_bin, matched, thresholds = [], [], {}
    for method, error in errors.items():
        reference = float(error[elevation < COPLANAR_ELEVATION_DEG].mean())
        thresholds[method] = dict(reference_mae_deg=reference, threshold_deg=None)
        for low, high in zip(ELEVATION_BINS_DEG[:-1], ELEVATION_BINS_DEG[1:]):
            for h in [None, *sorted(set(height))]:
                select = (elevation >= low) & (elevation < high) & (True if h is None else height == h)
                if not select.any():
                    continue
                mae, (ci_low, ci_high), _ = mean_interval(error[select], groups[select])
                by_bin.append(dict(method=method, elevation_low_deg=low, elevation_high_deg=high,
                                   height_difference_m="all" if h is None else h, samples=int(select.sum()),
                                   mae_deg=mae, mae_ci_low_deg=ci_low, mae_ci_high_deg=ci_high,
                                   ratio_to_coplanar=mae / reference))
                if (h is None and low >= COPLANAR_ELEVATION_DEG and mae > 2 * reference
                        and thresholds[method]["threshold_deg"] is None):
                    thresholds[method]["threshold_deg"] = low
        for d in sorted(set(distance[height == 2.])):
            if 2 * d not in set(distance):
                continue
            cells = {"h2_d": (height == 2.) & (distance == d), "h4_2d": (height == 4.) & (distance == 2 * d),
                     "h0_d": (height == 0.) & (distance == d), "h0_2d": (height == 0.) & (distance == 2 * d)}
            stats = {name: mean_interval(error[cell], groups[cell], seed=SEED + i)
                     for i, (name, cell) in enumerate(cells.items())}
            row = dict(method=method, distance_m=d, elevation_deg=float(np.rad2deg(np.arctan2(2., d))))
            for name, (mae, (ci_low, ci_high), _) in stats.items():
                row.update({f"{name}_mae_deg": mae, f"{name}_ci_low_deg": ci_low, f"{name}_ci_high_deg": ci_high})
            for label, a, b in (("h4_2d_minus_h2_d", "h4_2d", "h2_d"), ("h0_2d_minus_h0_d", "h0_2d", "h0_d")):
                low_diff, high_diff = percentile_interval(stats[a][2] - stats[b][2])
                row.update({f"{label}_deg": stats[a][0] - stats[b][0],
                            f"{label}_ci_low_deg": low_diff, f"{label}_ci_high_deg": high_diff})
            matched.append(row)
    write_csv(manifest_path.parent / "by_elevation_bin.csv", by_bin)
    write_csv(manifest_path.parent / "matched_elevation.csv", matched)
    _finish(manifest_path, thresholds=thresholds,
            threshold_rule="lowest bin (>= 10 deg) whose pooled MAE exceeds twice the MAE below 10 deg")
    return matched


def _circular_variance(angles_deg):
    return 1. - circular_summary(angles_deg)["resultant_length"]


def run_rotation(dataset: Path, output: Path) -> list[dict]:
    """T1.4: share of near-coplanar PDoA residual variance explained by tag rotation."""
    data, split, digest = _static_arrays(dataset, output)
    coplanar = data["elevation"] < COPLANAR_ELEVATION_DEG
    residual = wrap(data["phase"] - data["expected"])
    manifest_path = _start(output / "t1_4_rotation", source_sha256=digest, split=split, test="T1.4",
                           scope=f"train + validation, elevation < {COPLANAR_ELEVATION_DEG:g} deg",
                           explained="1 - weighted within-rotation circular variance / total circular variance")
    table = []
    for tag in sorted(set(data["tag"])):
        for h in [None, *sorted(set(data["height"]))]:
            select = coplanar & (data["tag"] == tag) & (True if h is None else data["height"] == h)
            if not select.any():
                continue
            values, rotations = residual[select], data["rotation"][select]
            means = {r: circular_summary(values[rotations == r])["circular_bias_deg"] for r in sorted(set(rotations))}
            spread = max(abs(float(wrap(a - b))) for a in means.values() for b in means.values())
            total = _circular_variance(values)
            within = sum((rotations == r).sum() * _circular_variance(values[rotations == r])
                         for r in means) / len(values)
            table.append(dict(tag=tag, height_difference_m="all" if h is None else h,
                              boresight_deg=float(data["bearing"][select][0] - 90.), samples=int(select.sum()),
                              **{f"rotation_{r:g}_mean_residual_deg": m for r, m in means.items()},
                              max_rotation_spread_deg=spread, total_circular_variance=total,
                              rotation_explained_share=1. - within / total if total > 0 else None))
    write_csv(manifest_path.parent / "by_tag_height.csv", table)
    _finish(manifest_path)
    return table


def run_session(dataset: Path, output: Path) -> list[dict]:
    """T1.5: do the three height sessions differ where elevation is negligible?

    Restricted to d >= 24 m, where every height is below 10 deg elevation, so
    the same distances and tags are compared across sessions.
    """
    data, split, digest = _static_arrays(dataset, output)
    far = data["distance"] >= SESSION_MIN_DISTANCE_M
    residual = wrap(data["phase"] - data["expected"])
    sessions = sorted(set(data["height"]))
    manifest_path = _start(output / "t1_5_session", source_sha256=digest, split=split, test="T1.5",
                           scope=f"train + validation, ground distance >= {SESSION_MIN_DISTANCE_M:g} m",
                           note="Each height was acquired in a separate session; height and session are confounded.",
                           bootstrap=dict(replicates=N_BOOT, seed=SEED, unit="configuration"))
    estimates = {}
    for i, h in enumerate(sessions):
        select = far & (data["height"] == h)
        estimates[("range_bias_m", "all", h)] = mean_interval(data["range_error"][select], data["group"][select],
                                                              seed=SEED + i)
        for tag in sorted(set(data["tag"])):
            cell = select & (data["tag"] == tag)
            if not cell.any():  # tag missing at these distances in this session
                continue
            estimates[("pdoa_offset_deg", tag, h)] = circular_mean_interval(residual[cell], data["group"][cell],
                                                                            seed=SEED + i)
    table = [dict(quantity=q, tag=t, height_difference_m=h, estimate=e, ci_low=ci[0], ci_high=ci[1])
             for (q, t, h), (e, ci, _) in estimates.items()]
    differences = []
    for quantity, tag in sorted({(q, t) for q, t, _ in estimates}):
        for a_index, a in enumerate(sessions):
            for b in sessions[a_index + 1:]:
                if (quantity, tag, a) not in estimates or (quantity, tag, b) not in estimates:
                    continue
                (eb, _, sb), (ea, _, sa) = estimates[(quantity, tag, b)], estimates[(quantity, tag, a)]
                if quantity == "pdoa_offset_deg":
                    estimate = float(wrap(eb - ea))
                    low, high = wrapped_interval(estimate, wrap(sb - sa))
                else:
                    estimate, (low, high) = eb - ea, percentile_interval(sb - sa)
                differences.append(dict(quantity=quantity, tag=tag, comparison=f"h{b:g}-h{a:g}",
                                        difference=estimate, ci_low=low, ci_high=high,
                                        excludes_zero=bool(low > 0 or high < 0)))
    write_csv(manifest_path.parent / "by_session.csv", table)
    write_csv(manifest_path.parent / "differences.csv", differences)
    _finish(manifest_path)
    return differences
