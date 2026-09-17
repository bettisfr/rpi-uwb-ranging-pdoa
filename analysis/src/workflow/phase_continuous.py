"""Continuous generic PDoA correction evaluated on an unseen bearing/tag."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .phase import circular_summary, fit_phase_offset, wrap
from .phase_trigonometric import fit_trigonometric, predict_correction
from .offset import load_assigned_rows


def base_features(phase_deg, range_m, height_m):
    """Observed continuous features; no tag, bearing, or orientation label."""
    phase = np.deg2rad(np.asarray(phase_deg, dtype=float))
    return np.column_stack((np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase),
                            np.asarray(range_m, dtype=float), np.asarray(height_m, dtype=float)))


def harmonic_design(features, center, scale):
    """Fixed low-order continuous basis with range/height interactions."""
    s1, c1, s2, c2 = features[:, :4].T
    zr, zh = ((features[:, 4:] - center) / scale).T
    return np.column_stack((np.ones(len(features)), s1, c1, s2, c2, zr, zh,
                            zr*s1, zr*c1, zh*s1, zh*c1))


def fit_harmonic(features, expected_deg):
    center, scale = features[:, 4:].mean(axis=0), features[:, 4:].std(axis=0)
    if np.any(scale == 0):
        raise ValueError("Range/height features need nonzero training variance")
    X = harmonic_design(features, center, scale)
    radians = np.deg2rad(expected_deg)
    target = np.column_stack((np.cos(radians), np.sin(radians)))
    coefficients, _, rank, _ = np.linalg.lstsq(X, target, rcond=None)
    if rank != X.shape[1]:
        raise ValueError("Continuous harmonic design is rank deficient")
    return dict(center=center, scale=scale, coefficients=coefficients)


def predict_harmonic(features, model):
    vector = harmonic_design(features, model["center"], model["scale"]) @ model["coefficients"]
    return np.rad2deg(np.arctan2(vector[:, 1], vector[:, 0]))


def fit_torch_mlp(features_train, expected_train, features_eval, *, seed=42):
    """Predict expected phase with the common fully CUDA-resident MLP routine.

    Keeping the complete fold tensors on the GPU is important here: LOBO trains
    nine large development folds, so a CPU DataLoader and per-batch host/device
    transfers would otherwise dominate the runtime.
    """
    from .phase_torch import PARAMETERS, fit_predict

    if seed != PARAMETERS["seed"]:
        raise ValueError("The continuous benchmark uses the shared fixed CUDA seed")
    radians = np.deg2rad(expected_train)
    train_y = np.column_stack((np.cos(radians), np.sin(radians)))
    vector, _, info = fit_predict(features_train, train_y, features_eval)
    return np.rad2deg(np.arctan2(vector[:, 1], vector[:, 0])), {
        "runtime": "shared fully CUDA-resident PyTorch implementation",
        "epochs": PARAMETERS["epochs"],
        "batch_size": PARAMETERS["batch_size"],
        **info,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_continuous_lobo(dataset: Path, output: Path) -> list[dict]:
    """Leave one fixed bearing/tag out while fitting only generic features."""
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in ("train", "validation")]
    phase = np.asarray([float(row["pdoa_deg"]) for row in rows])
    expected = np.asarray([float(row["eta_expected_deg"]) for row in rows])
    range_m = np.asarray([float(row["range_m"]) for row in rows])
    height_m = np.asarray([float(row["height_difference_m"]) for row in rows])
    tags = np.asarray([row["tag"] for row in rows])
    bearings = {row["tag"]: float(row["gt_bearing_deg"]) for row in rows}
    features = base_features(phase, range_m, height_m)
    if not np.isfinite(features).all() or not np.isfinite(expected).all():
        raise ValueError("Continuous PDoA benchmark needs finite inputs and targets")
    destination = output / "pdoa_continuous_lobo"
    destination.mkdir(exist_ok=True)
    manifest = dict(status="running", protocol="leave_one_bearing_out",
                    source_sha256=digest, split=split,
                    input_features=["sin/cos(pdoa_deg)", "sin/cos(2*pdoa_deg)", "range_m", "height_difference_m"],
                    forbidden_features=["tag", "bearing", "tag_rotation_deg", "ground-truth position"],
                    models=["raw", "trigonometric_global", "harmonic_range_height", "mlp_range_height_cuda"],
                    development_pool="original train + validation; original grouped test excluded",
                    caveat="Each held tag is also one held bearing in the present static deployment.",
                    folds=[])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pooled = {name: [] for name in manifest["models"]}
    metrics, predictions = [], []
    for tag in sorted(set(tags)):
        train, evaluate = tags != tag, tags == tag
        # Existing global periodic residual baseline, fitted only on other bearings.
        subset = np.full(train.sum(), "train", dtype=object)
        trig = fit_trigonometric(phase[train], expected[train], subset)
        trig_correction, _, _ = predict_correction(phase[evaluate], trig)
        harmonic = fit_harmonic(features[train], expected[train])
        estimates = {
            "raw": wrap(phase[evaluate]),
            "trigonometric_global": wrap(phase[evaluate] - trig_correction),
            "harmonic_range_height": wrap(predict_harmonic(features[evaluate], harmonic)),
        }
        mlp_prediction, mlp_info = fit_torch_mlp(features[train], expected[train], features[evaluate])
        estimates["mlp_range_height_cuda"] = wrap(mlp_prediction)
        expected_eval = expected[evaluate]
        fold_id = f"bearing_{bearings[tag]:g}deg_{tag}"
        fold = dict(fold_id=fold_id, held_tag=tag, held_bearing_deg=bearings[tag],
                    training_samples=int(train.sum()), evaluation_samples=int(evaluate.sum()), mlp=mlp_info)
        manifest["folds"].append(fold)
        for method, predicted in estimates.items():
            error = wrap(predicted - expected_eval)
            pooled[method].append(error)
            metrics.append(dict(fold_id=fold_id, held_tag=tag, held_bearing_deg=bearings[tag], method=method,
                                **circular_summary(error)))
        for position, index in enumerate(np.flatnonzero(evaluate)):
            predictions.append(dict(csv_line=rows[index]["csv_line"], held_tag=tag, held_bearing_deg=bearings[tag],
                                    expected_phase_deg=expected[index],
                                    **{f"{method}_phase_deg": values[position] for method, values in estimates.items()}))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Continuous PDoA LOBO {tag} ({bearings[tag]:g} deg) complete", flush=True)
    overall = [dict(method=method, **circular_summary(np.concatenate(errors))) for method, errors in pooled.items()]
    write_csv(destination / "metrics.csv", metrics)
    write_csv(destination / "overall_metrics.csv", overall)
    write_csv(destination / "predictions.csv", predictions)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return overall
