"""CUDA MLP comparison for circular PDoA residual correction."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from .linear import write_csv
from .offset import load_assigned_rows
from .phase import circular_summary, fit_phase_offset, wrap
from .phase_learning import phase_features
from .phase_trigonometric import fit_trigonometric, predict_correction


PARAMETERS = {
    "hidden_layers": [64, 32],
    "batch_size": 4096,
    "epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "seed": 42,
}


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch CUDA is not installed; install analysis/requirements-gpu.txt") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA device")
    return torch


def fit_predict(X_train, y_train, X_evaluate):
    """Train a fixed MLP on CUDA, using training data for every transform."""
    torch = _torch()
    torch.manual_seed(PARAMETERS["seed"])
    torch.cuda.manual_seed_all(PARAMETERS["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_evaluate = torch.tensor(X_evaluate, dtype=torch.float32, device=device)
    x_mean = X_train.mean(0)
    x_std = X_train.std(0, correction=0).clamp_min(1e-8)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0, correction=0).clamp_min(1e-8)
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], PARAMETERS["hidden_layers"][0]),
        torch.nn.ReLU(),
        torch.nn.Linear(PARAMETERS["hidden_layers"][0], PARAMETERS["hidden_layers"][1]),
        torch.nn.ReLU(),
        torch.nn.Linear(PARAMETERS["hidden_layers"][1], 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMETERS["learning_rate"],
                                  weight_decay=PARAMETERS["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(PARAMETERS["seed"])
    initial_loss = final_loss = None
    start = time.monotonic()
    for epoch in range(PARAMETERS["epochs"]):
        permutation = torch.randperm(len(X_train), generator=generator, device=device)
        total = 0.0
        for start_index in range(0, len(X_train), PARAMETERS["batch_size"]):
            index = permutation[start_index:start_index + PARAMETERS["batch_size"]]
            prediction = model((X_train[index]-x_mean)/x_std)
            loss = torch.nn.functional.mse_loss(prediction, (y_train[index]-y_mean)/y_std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        final_loss = total / len(X_train)
        if epoch == 0:
            initial_loss = final_loss
    with torch.no_grad():
        vectors = (model((X_evaluate-x_mean)/x_std)*y_std+y_mean).cpu().numpy()
    torch.cuda.synchronize()
    artifact = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "x_mean": x_mean.cpu(), "x_std": x_std.cpu(), "y_mean": y_mean.cpu(), "y_std": y_std.cpu(),
        "parameters": PARAMETERS, "input_dimensions": int(X_train.shape[1]),
    }
    info = {
        "elapsed_s": time.monotonic()-start, "initial_training_loss": initial_loss,
        "final_training_loss": final_loss, "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
    }
    return vectors, artifact, info


def run_torch_phase_learning(dataset: Path, output: Path):
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [row for row in rows if row["subset"] in ("train", "validation")]
    subset = np.array([row["subset"] for row in rows])
    train, validation = subset == "train", subset == "validation"
    phase = np.array([float(row["pdoa_deg"]) for row in rows])
    expected = np.array([float(row["eta_expected_deg"]) for row in rows])
    target = np.column_stack((np.cos(np.deg2rad(wrap(phase-expected))),
                              np.sin(np.deg2rad(wrap(phase-expected)))))
    offset = fit_phase_offset(phase, expected, subset)
    trigonometric = fit_trigonometric(phase, expected, subset)
    correction, _, _ = predict_correction(phase[validation], trigonometric)
    estimates = {
        "raw": wrap(phase[validation]),
        "circular_offset": wrap(phase[validation]-offset),
        "trigonometric": wrap(phase[validation]-correction),
    }
    destination = output / "pdoa_torch_gpu"
    destination.mkdir(exist_ok=True)
    manifest = {
        "status": "running", "source_sha256": digest, "split": split,
        "target": "cos/sin(wrap(measured_phase-expected_phase))", "parameters": PARAMETERS,
        "training_samples": int(train.sum()), "validation_samples": int(validation.sum()),
        "outlier_policy": "All finite phase records retained; no filtering or clipping",
        "protocol": "Fit and standardize on training only; evaluate unchanged validation; test excluded",
        "models": [],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    torch = _torch()
    for enriched in (False, True):
        X, feature_names = phase_features(rows, enriched)
        name = "torch_mlp_enriched" if enriched else "torch_mlp_phase_only"
        print(f"Fitting {name} on CUDA", flush=True)
        vectors, artifact, info = fit_predict(X[train], target[train], X[validation])
        correction = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
        norm = np.linalg.norm(vectors, axis=1)
        fallback = norm < 1e-8
        correction[fallback] = offset
        estimates[name] = wrap(phase[validation]-correction)
        torch.save(artifact, destination / f"{name}.pt")
        manifest["models"].append({"name": name, "feature_names": feature_names,
                                   "fallback_count": int(fallback.sum()), "artifact": f"{name}.pt", **info})
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    heights = np.array([float(row["height_difference_m"]) for row in rows])[validation]
    tags = np.array([row["tag"] for row in rows])[validation]
    summary, details = [], []
    for name, estimate in estimates.items():
        errors = wrap(estimate-expected[validation])
        for height in [None, *sorted(set(heights))]:
            selected = np.ones(len(heights), dtype=bool) if height is None else heights == height
            summary.append({"subset": "validation", "method": name,
                            "height_difference_m": "all" if height is None else height,
                            **circular_summary(errors[selected])})
            for tag in sorted(set(tags)):
                select_tag = selected & (tags == tag)
                if select_tag.any():
                    details.append({"subset": "validation", "method": name,
                                    "height_difference_m": "all" if height is None else height,
                                    "tag": tag, **circular_summary(errors[select_tag])})
    prediction_rows = []
    for position, row_index in enumerate(np.flatnonzero(validation)):
        prediction_rows.append({"csv_line": rows[row_index]["csv_line"], "group_id": rows[row_index]["group_id"],
                                "tag": tags[position], "height_difference_m": heights[position],
                                "expected_phase_deg": expected[row_index],
                                **{name+"_deg": values[position] for name, values in estimates.items()}})
    write_csv(destination / "predictions.csv", prediction_rows)
    write_csv(destination / "metrics.csv", summary)
    write_csv(destination / "metrics_by_tag_height.csv", details)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return summary
