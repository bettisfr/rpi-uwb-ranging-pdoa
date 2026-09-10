"""Leave-one-distance-out PDoA correction by bearing and tag rotation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .phase import circular_summary, fit_phase_offset, wrap
from .phase_trigonometric import fit_trigonometric, predict_correction
from .offset import load_assigned_rows


STRATEGIES = ("global", "bearing", "rotation", "bearing_rotation")


def strategy_key(rows, strategy):
    """Return a known deployment condition, never a ground-truth coordinate."""
    if strategy == "global":
        return np.full(len(rows), "global", dtype=object)
    if strategy == "bearing":
        # In this deployment each tag occupies one fixed surveyed bearing.
        return np.asarray([row["tag"] for row in rows], dtype=object)
    if strategy == "rotation":
        return np.asarray([row["tag_rotation_deg"] for row in rows], dtype=object)
    if strategy == "bearing_rotation":
        return np.asarray([f"{row['tag']}|{row['tag_rotation_deg']}" for row in rows], dtype=object)
    raise ValueError(f"Unknown PDoA stratification strategy: {strategy}")


def fit_and_predict(phase, expected, train, evaluate, keys, strategy):
    """Fit offset and periodic corrections within each requested stratum."""
    offset_prediction = np.empty(evaluate.sum())
    trig_prediction = np.empty(evaluate.sum())
    models = []
    positions = np.flatnonzero(evaluate)
    for key in sorted(set(keys[evaluate]), key=str):
        train_group = train & (keys == key)
        evaluate_group = evaluate & (keys == key)
        if not train_group.any():
            raise ValueError(f"No training rows for strategy={strategy}, group={key!r}")
        # Reuse the existing circular/trigonometric definitions with an
        # artificial all-train subset limited to this physical condition.
        group_phase, group_expected = phase[train_group], expected[train_group]
        subset = np.full(len(group_phase), "train", dtype=object)
        offset = fit_phase_offset(group_phase, group_expected, subset)
        trig = fit_trigonometric(group_phase, group_expected, subset)
        prediction, magnitude, fallback = predict_correction(phase[evaluate_group], trig)
        local = np.searchsorted(positions, np.flatnonzero(evaluate_group))
        offset_prediction[local] = wrap(phase[evaluate_group] - offset)
        trig_prediction[local] = wrap(phase[evaluate_group] - prediction)
        models.append(dict(group=str(key), training_samples=int(train_group.sum()),
                           offset_deg=float(offset), trig_fallback_count=int(fallback.sum()),
                           trig_minimum_vector_norm=trig["minimum_vector_norm"]))
    return offset_prediction, trig_prediction, models


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stratified_lodo(dataset: Path, output: Path) -> list[dict]:
    """Compare global, bearing, rotation, and joint PDoA corrections fairly."""
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in ("train", "validation")]
    phase = np.asarray([float(row["pdoa_deg"]) for row in rows])
    expected = np.asarray([float(row["eta_expected_deg"]) for row in rows])
    distance = np.asarray([float(row["target_ground_distance_m"]) for row in rows])
    if not np.isfinite(phase).all() or not np.isfinite(expected).all():
        raise ValueError("PDoA benchmark needs finite measured and expected phase")
    destination = output / "pdoa_stratified_lodo"
    destination.mkdir(exist_ok=True)
    manifest = dict(status="running", protocol="leave_one_distance_out",
                    source_sha256=digest, split=split,
                    target="eta_expected_deg = -180 * gt_x_m / gt_range_3d_m",
                    input="pdoa_deg; known tag identity and known tag rotation only",
                    strategies=list(STRATEGIES),
                    models=["circular_offset", "trigonometric_two_harmonic"],
                    development_pool="original train + validation; original grouped test excluded",
                    outlier_policy="All finite phase records retained; no range-quality filter",
                    note="Bearing is operationally equivalent to tag identity in this fixed deployment.",
                    folds=[])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metrics, predictions = [], []
    pooled = {"raw": []}
    for strategy in STRATEGIES:
        pooled[f"offset_{strategy}"] = []
        pooled[f"trigonometric_{strategy}"] = []
    development = np.ones(len(rows), dtype=bool)
    for held_distance in sorted(set(distance)):
        train = development & (distance != held_distance)
        evaluate = development & (distance == held_distance)
        expected_eval = expected[evaluate]
        indices = np.flatnonzero(evaluate)
        estimates = {"raw": wrap(phase[evaluate])}
        fold = dict(held_distance_m=float(held_distance), training_samples=int(train.sum()),
                    evaluation_samples=int(evaluate.sum()), strategies={})
        for strategy in STRATEGIES:
            keys = strategy_key(rows, strategy)
            offset, trig, models = fit_and_predict(phase, expected, train, evaluate, keys, strategy)
            estimates[f"offset_{strategy}"] = offset
            estimates[f"trigonometric_{strategy}"] = trig
            fold["strategies"][strategy] = models
        for method, estimate in estimates.items():
            errors = wrap(estimate - expected_eval)
            pooled[method].append(errors)
            metrics.append(dict(fold_id=f"distance_{held_distance:g}m", held_distance_m=held_distance,
                                method=method, **circular_summary(errors)))
        for position, index in enumerate(indices):
            predictions.append(dict(csv_line=rows[index]["csv_line"], tag=rows[index]["tag"],
                                    height_difference_m=rows[index]["height_difference_m"],
                                    tag_rotation_deg=rows[index]["tag_rotation_deg"],
                                    held_distance_m=held_distance, expected_phase_deg=expected[index],
                                    **{f"{name}_phase_deg": value[position]
                                       for name, value in estimates.items()}))
        manifest["folds"].append(fold)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"PDoA LODO {held_distance:g} m complete", flush=True)
    overall = [dict(method=name, **circular_summary(np.concatenate(errors)))
               for name, errors in pooled.items()]
    write_csv(destination / "metrics.csv", metrics)
    write_csv(destination / "overall_metrics.csv", overall)
    write_csv(destination / "predictions.csv", predictions)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return overall
