"""Informative sector-classifier plus bearing-expert PDoA position benchmark."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .phase import circular_summary, fit_phase_offset, wrap
from .phase_trigonometric import fit_trigonometric, predict_correction
from .position_benchmark import metrics, reconstruct, write_csv
from .offset import load_assigned_rows


def features(phase, range_m, height):
    phase = np.deg2rad(phase)
    return np.column_stack((np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase), range_m, height))


def range_predictions(directory):
    result = {}
    for path in directory.glob("distance_*m/predictions.csv"):
        with path.open(newline="") as stream:
            result.update({int(r["csv_line"]): r for r in csv.DictReader(stream)})
    return result


def run_sector_position_lodo(dataset: Path, output: Path):
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [r for r in rows if r["subset"] in ("train", "validation") and r["ground_distance_error_m"] != "" and abs(float(r["ground_distance_error_m"])) <= 3]
    phase = np.array([float(r["pdoa_deg"]) for r in rows]); expected = np.array([float(r["eta_expected_deg"]) for r in rows])
    distance = np.array([float(r["target_ground_distance_m"]) for r in rows]); height = np.array([float(r["height_difference_m"]) for r in rows])
    tags = np.array([r["tag"] for r in rows]); range_m = np.array([float(r["range_m"]) for r in rows])
    X = features(phase, range_m, height)
    ranges = range_predictions(output / "distance_lodo")
    destination = output / "position_sector_lodo"; destination.mkdir(exist_ok=True)
    manifest = dict(status="running", source_sha256=digest, split=split, protocol="leave_one_distance_out",
                    classifier="RandomForestClassifier on sin/cos(PDoA), measured range, and height",
                    forbidden_features=["tag", "bearing", "rotation", "ground-truth position"],
                    expert="two-harmonic PDoA correction per bearing", note="Informative fixed-sector study; not continuous-bearing localization.")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    records=[]; phase_errors={"raw":[],"sector":[],"oracle":[]}; position_values={"sector":[],"oracle":[]}; accuracy=[]
    for held in sorted(set(distance)):
        train, evaluate = distance != held, distance == held
        classifier = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=10, random_state=42, n_jobs=2)
        classifier.fit(X[train], tags[train]); selected = classifier.predict(X[evaluate]); accuracy.append((selected == tags[evaluate]).mean())
        experts={}
        for tag in sorted(set(tags)):
            use = train & (tags == tag); experts[tag] = fit_trigonometric(phase[use], expected[use], np.full(use.sum(), "train", dtype=object))
        predicted_phase=np.empty(evaluate.sum()); oracle_phase=np.empty(evaluate.sum())
        for tag in sorted(set(tags)):
            for mask, target in ((selected == tag, predicted_phase), (tags[evaluate] == tag, oracle_phase)):
                if mask.any():
                    correction, _, _ = predict_correction(phase[evaluate][mask], experts[tag]); target[mask] = wrap(phase[evaluate][mask]-correction)
        rho=np.array([float(ranges[int(rows[i]["csv_line"])]["quadratic_regression_m"]) for i in np.flatnonzero(evaluate)])
        sx,sy,sc=reconstruct(rho,height[evaluate],predicted_phase); ox,oy,oc=reconstruct(rho,height[evaluate],oracle_phase)
        xt=np.array([float(rows[i]["gt_x_m"]) for i in np.flatnonzero(evaluate)]); yt=np.array([float(rows[i]["gt_y_m"]) for i in np.flatnonzero(evaluate)])
        phase_errors["raw"].append(wrap(phase[evaluate]-expected[evaluate])); phase_errors["sector"].append(wrap(predicted_phase-expected[evaluate])); phase_errors["oracle"].append(wrap(oracle_phase-expected[evaluate]))
        position_values["sector"].append((xt,yt,sx,sy,sc)); position_values["oracle"].append((xt,yt,ox,oy,oc))
        for k, i in enumerate(np.flatnonzero(evaluate)):
            records.append(dict(csv_line=rows[i]["csv_line"],held_distance_m=held,true_tag=tags[i],predicted_tag=selected[k],classified_correct=bool(selected[k]==tags[i]),x_true_m=xt[k],y_true_m=yt[k],sector_x_m=sx[k],sector_y_m=sy[k],oracle_x_m=ox[k],oracle_y_m=oy[k]))
        print(f"Sector LODO {held:g} m: classification accuracy {(selected == tags[evaluate]).mean():.3f}", flush=True)
    phase_summary=[dict(method=name,**circular_summary(np.concatenate(values))) for name, values in phase_errors.items()]
    position_summary=[]
    for name, values in position_values.items():
        xt,yt,xh,yh,clipped=(np.concatenate([v[i] for v in values]) for i in range(5)); position_summary.append(dict(method=name,**metrics(xt,yt,xh,yh,clipped)))
    write_csv(destination / "predictions.csv",records); write_csv(destination / "phase_metrics.csv",phase_summary); write_csv(destination / "position_metrics.csv",position_summary)
    manifest.update(status="complete",classification_accuracy=float(np.mean(accuracy)),classification_accuracy_by_fold=[float(x) for x in accuracy])
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return position_summary, float(np.mean(accuracy))
