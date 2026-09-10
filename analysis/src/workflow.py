#!/usr/bin/env python3
"""Prepare reproducible benchmark contracts for PDoA correction experiments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from workflow.contracts import MODEL_CATALOG, SPLIT_DESCRIPTIONS
from workflow.audit import audit_static
from workflow.offset import run_offset
from workflow.linear import run_linear
from workflow.export import prepare_output_contracts
from workflow.features import validate_feature_set
from workflow.splits import export_grouped_split, validate_split_name


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
DEFAULT_CONFIG = ANALYSIS / "configs" / "static_benchmark.json"
OUTPUTS = ANALYSIS / "outputs"


def load_config(path: Path) -> dict[str, object]:
    """Load and validate the versioned static benchmark definition."""
    with path.open() as source:
        config = json.load(source)
    for name, features in config["feature_sets"].items():
        if not isinstance(features, list):
            raise ValueError(f"feature set {name!r} must be a list")
        validate_feature_set(features)
    for split in config["splits"]:
        validate_split_name(split)
    unknown_models = set(config["models"]) - set(MODEL_CATALOG)
    if unknown_models:
        raise ValueError(f"unknown model(s): {sorted(unknown_models)}")
    return config


def write_plan(config: dict[str, object], output_directory: Path) -> Path:
    """Export one planned benchmark combination per row for review and plotting."""
    plan_path = output_directory / "benchmark_plan.csv"
    with plan_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("campaign", "feature_set", "target_name", "model_name", "model_stage", "model_status", "split_name", "split_description"),
        )
        writer.writeheader()
        for feature_set in config["feature_sets"]:
            for target in config["targets"]:
                for model_name in config["models"]:
                    model = MODEL_CATALOG[model_name]
                    for split_name in config["splits"]:
                        writer.writerow(
                            {
                                "campaign": config["campaign"],
                                "feature_set": feature_set,
                                "target_name": target,
                                "model_name": model_name,
                                "model_stage": model.stage,
                                "model_status": model.status,
                                "split_name": split_name,
                                "split_description": SPLIT_DESCRIPTIONS[split_name],
                            }
                        )
    return plan_path


def write_manifest(config_path: Path, config: dict[str, object], output_directory: Path) -> Path:
    """Record immutable benchmark intent next to the generated CSV contracts."""
    manifest_path = output_directory / "benchmark_manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "dataset": config["dataset"],
        "status": "skeleton_only",
        "note": "No measurements were fitted and no model predictions were generated.",
    }
    with manifest_path.open("w") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare", action="store_true", help="create stable CSV output contracts")
    parser.add_argument("--plan", action="store_true", help="export the planned benchmark combinations")
    parser.add_argument("--audit-static", action="store_true", help="validate static rows and export configuration/run inventories")
    parser.add_argument("--split-static", action="store_true", help="audit and export grouped train/validation/test assignments")
    parser.add_argument("--fit-offset", action="store_true", help="audit, split, and evaluate training-only constant distance correction")
    parser.add_argument("--fit-linear", action="store_true", help="use existing grouped split to fit OLS and compare on validation")
    parser.add_argument("--fit-quadratic", action="store_true", help="compare quadratic with raw, offset and linear on validation")
    parser.add_argument("--fit-forest", action="store_true", help="compare fixed random forest with parametric baselines on validation")
    parser.add_argument("--compare-distance", action="store_true", help="compare all distance models including SVR and MLP on validation")
    parser.add_argument("--lodo-distance", action="store_true", help="evaluate fixed distance models leaving one distance out of the development pool")
    parser.add_argument("--fit-phase-offset", action="store_true", help="fit circular PDoA offset and evaluate train/validation")
    parser.add_argument("--fit-phase-trig", action="store_true", help="fit periodic phase correction and compare on validation")
    parser.add_argument("--compare-phase", action="store_true", help="compare RF/MLP with phase-only and enriched features")
    parser.add_argument("--fit-phase-svr", action="store_true", help="try phase SVR with phase-only and enriched inputs")
    parser.add_argument("--fit-phase-torch", action="store_true", help="train CUDA PyTorch phase MLPs and compare on validation")
    parser.add_argument("--lodo-phase-stratified", action="store_true", help="evaluate global, bearing, rotation, and joint PDoA corrections")
    parser.add_argument("--evaluate-generic-position", action="store_true", help="evaluate generic out-of-fold geometric position reconstruction")
    parser.add_argument("--lobo-phase-continuous", action="store_true", help="test continuous generic PDoA models on a held-out bearing")
    parser.add_argument("--evaluate-sector-position", action="store_true", help="informative sector classifier plus bearing-expert position evaluation")
    args = parser.parse_args()
    if not any((args.prepare, args.plan, args.audit_static, args.split_static, args.fit_offset,
                args.fit_linear, args.fit_quadratic, args.fit_forest, args.compare_distance, args.lodo_distance, args.fit_phase_offset, args.fit_phase_trig, args.compare_phase, args.fit_phase_svr, args.fit_phase_torch, args.lodo_phase_stratified, args.evaluate_generic_position, args.lobo_phase_continuous, args.evaluate_sector_position)):
        parser.error("choose an action; see --help")

    config_path = args.config.resolve()
    config = load_config(config_path)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if args.fit_phase_torch:
        from workflow.phase_torch import run_torch_phase_learning
        for row in run_torch_phase_learning(ROOT / config['dataset'], OUTPUTS):
            if row['height_difference_m'] == 'all':
                print(f"{row['method']}: MAE={row['mae_deg']:.4f} deg, RMSE={row['rmse_deg']:.4f} deg")
    if args.lodo_phase_stratified:
        from workflow.phase_stratified import run_stratified_lodo
        for row in run_stratified_lodo(ROOT / config['dataset'], OUTPUTS):
            print(f"PDoA LODO {row['method']}: MAE={row['mae_deg']:.4f} deg, RMSE={row['rmse_deg']:.4f} deg, P90={row['p90_absolute_error_deg']:.4f} deg")
    if args.evaluate_generic_position:
        from workflow.position_benchmark import run_generic_position_benchmark
        for row in run_generic_position_benchmark(ROOT / config['dataset'], OUTPUTS):
            print(f"Position LODO {row['method']}: MAE={row['position_mae_m']:.4f} m, RMSE={row['position_rmse_m']:.4f} m, P90={row['position_p90_m']:.4f} m")
    if args.lobo_phase_continuous:
        from workflow.phase_continuous import run_continuous_lobo
        for row in run_continuous_lobo(ROOT / config['dataset'], OUTPUTS):
            print(f"Continuous PDoA LOBO {row['method']}: MAE={row['mae_deg']:.4f} deg, RMSE={row['rmse_deg']:.4f} deg, P90={row['p90_absolute_error_deg']:.4f} deg")
    if args.evaluate_sector_position:
        from workflow.sector_position import run_sector_position_lodo
        rows, accuracy = run_sector_position_lodo(ROOT / config['dataset'], OUTPUTS)
        print(f"Sector classifier LODO accuracy={accuracy:.4f}")
        for row in rows:
            if row['kind'] == 'position': print(f"Sector position {row['method']}: MAE={row['position_mae_m']:.4f} m, RMSE={row['position_rmse_m']:.4f} m")
    if args.fit_phase_svr:
        from workflow.phase_learning import run_phase_learning
        for row in run_phase_learning(ROOT / config['dataset'], OUTPUTS, svr_only=True):
            if row['height_difference_m'] == 'all':
                print(f"{row['method']}: MAE={row['mae_deg']:.4f} deg, RMSE={row['rmse_deg']:.4f} deg")
    if args.compare_phase:
        from workflow.phase_learning import run_phase_learning
        for row in run_phase_learning(ROOT / config['dataset'], OUTPUTS):
            if row['height_difference_m'] == 'all':
                print(f"{row['method']}: MAE={row['mae_deg']:.4f} deg, RMSE={row['rmse_deg']:.4f} deg")
    if args.fit_phase_trig:
        from workflow.phase import run_phase
        model, results = run_phase(ROOT / config['dataset'], OUTPUTS, trigonometric=True)
        for row in results:
            if row['subset'] == 'validation' and row['height_difference_m'] == 'all':
                print(f"Validation {row['method']}: MAE={row['mae_deg']:.6f} deg, RMSE={row['rmse_deg']:.6f} deg, P90={row['p90_absolute_error_deg']:.6f} deg")
    if args.fit_phase_offset:
        from workflow.phase import run_phase
        model, results = run_phase(ROOT / config['dataset'], OUTPUTS)
        print(f"Training phase offset: {model['offset_deg']:.6f} deg")
        for row in results:
            if row['subset'] == 'validation' and row['height_difference_m'] == 'all':
                print(f"Validation {row['method']}: MAE={row['mae_deg']:.6f} deg, RMSE={row['rmse_deg']:.6f} deg")
    if args.audit_static or args.split_static or args.fit_offset:
        report = audit_static(ROOT / config["dataset"], OUTPUTS)
        print(f"Static audit: {report['status']}; {report['rows']} rows, {report['runs']} runs, "
              f"{report['observed_configurations']} configurations")
        print(f"Report and inventories: {OUTPUTS}")
        if report["status"] != "passed":
            raise SystemExit("Static audit failed; inspect static_audit.json before proceeding.")
        if args.split_static or args.fit_offset:
            summary = export_grouped_split(OUTPUTS, report, config["grouped_split"])
            for row in summary:
                if row["height_difference_m"] == "all":
                    print(f"{row['subset']}: {row['groups']} groups, {row['samples']} samples")
    if args.fit_offset:
        model, results = run_offset(ROOT / config['dataset'], OUTPUTS)
        print(f"Training offset: {model['offset_m']:.6f} m")
        for row in results:
            if row['subset'] == 'test' and row['height_difference_m'] == 'all':
                print(f"Test {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m")
    if args.prepare:
        for path in prepare_output_contracts(OUTPUTS):
            print(f"Prepared {path.relative_to(ROOT)}")
    if args.fit_linear:
        model, results = run_linear(ROOT / config['dataset'], OUTPUTS)
        print(f"Linear: slope={model['slope']:.8f}, intercept={model['intercept_m']:.8f} m")
        for row in results:
            if row['subset'] == 'validation' and row['height_difference_m'] == 'all':
                print(f"Validation {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m")
    if args.plan:
        print(f"Wrote {write_plan(config, OUTPUTS).relative_to(ROOT)}")
        print(f"Wrote {write_manifest(config_path, config, OUTPUTS).relative_to(ROOT)}")
    if args.fit_quadratic:
        model, results = run_linear(ROOT / config['dataset'], OUTPUTS, include_quadratic=True)
        print(f"Quadratic coefficients: {model['quadratic']}")
        for row in results:
            if row['subset'] == 'validation' and row['height_difference_m'] == 'all':
                print(f"Validation {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m, P90={row['p90_absolute_error_m']:.6f} m")
    if args.fit_forest:
        model, results = run_linear(ROOT / config['dataset'], OUTPUTS,
                                    include_quadratic=True, include_forest=True)
        for row in results:
            if row['subset'] == 'validation' and row['height_difference_m'] == 'all':
                print(f"Validation {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m, P90={row['p90_absolute_error_m']:.6f} m")
    if args.compare_distance or args.lodo_distance:
        from workflow.distance_benchmark import run_benchmark
        for enabled, protocol in ((args.compare_distance, 'grouped'), (args.lodo_distance, 'lodo')):
            if enabled:
                results = run_benchmark(ROOT / config['dataset'], OUTPUTS,
                                        ANALYSIS / 'configs/distance_models.json', protocol)
                for row in results:
                    print(f"{protocol} {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m, P90={row['p90_absolute_error_m']:.6f} m")


if __name__ == "__main__":
    main()
