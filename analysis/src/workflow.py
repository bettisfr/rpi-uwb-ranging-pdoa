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
    parser.add_argument("--t0-firmware-frame", action="store_true", help="T0.1: test whether firmware x/y error is a frame convention")
    parser.add_argument("--t0-mlp-check", action="store_true", help="T0.2: separate MLP fitting bug from bearing extrapolation")
    parser.add_argument("--t0-bearing-label", action="store_true", help="T0.3: compare per-bearing PDoA correction with a label-only constant")
    parser.add_argument("--t1-characteristic", action="store_true", help="T1.1: fit PDoA = wrap(scale * expected + offset) at low elevation")
    parser.add_argument("--t1-invertibility", action="store_true", help="T1.2: per-bearing invertibility and useful sector")
    parser.add_argument("--t1-elevation", action="store_true", help="T1.3: PDoA error by elevation, matched h=2/h=4 comparison")
    parser.add_argument("--t1-rotation", action="store_true", help="T1.4: residual variance explained by tag rotation")
    parser.add_argument("--t1-session", action="store_true", help="T1.5: compare sessions/heights at negligible elevation")
    parser.add_argument("--phase1", action="store_true", help="run T1.1-T1.5")
    parser.add_argument("--loho-distance", action="store_true", help="T2.1: distance models leaving one height out")
    parser.add_argument("--pdoa-elevation", choices=("lodo", "loho", "lobo"), action="append",
                        help="T2.2: generic PDoA models incl. elevation/range under a held-out protocol (repeatable)")
    parser.add_argument("--pdoa-gate", action="store_true", help="T2.3: observable-only reliability gate on LODO predictions")
    parser.add_argument("--position-elevation", choices=("lodo", "loho"), action="append",
                        help="T2.4: position from out-of-fold range and PDoA, with and without gate (repeatable)")
    parser.add_argument("--phase2", action="store_true", help="run T2.1-T2.4 in dependency order")
    parser.add_argument("--final-test", action="store_true", help="T3.2: one-time evaluation of frozen models on the grouped test")
    parser.add_argument("--force-final-test", action="store_true", help="rerun T3.2 after a crash (recorded in the manifest)")
    args = parser.parse_args()
    if not any((args.prepare, args.plan, args.audit_static, args.split_static, args.fit_offset,
                args.fit_linear, args.fit_quadratic, args.fit_forest, args.compare_distance, args.lodo_distance, args.fit_phase_offset, args.fit_phase_trig, args.compare_phase, args.fit_phase_svr, args.fit_phase_torch, args.lodo_phase_stratified, args.evaluate_generic_position, args.lobo_phase_continuous, args.evaluate_sector_position,
                args.t0_firmware_frame, args.t0_mlp_check, args.t0_bearing_label,
                args.t1_characteristic, args.t1_invertibility, args.t1_elevation, args.t1_rotation,
                args.t1_session, args.phase1, args.loho_distance, args.pdoa_elevation,
                args.pdoa_gate, args.position_elevation, args.phase2, args.final_test)):
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
            print(f"Sector position {row['method']}: MAE={row['position_mae_m']:.4f} m, RMSE={row['position_rmse_m']:.4f} m")
    if args.t0_firmware_frame:
        from workflow.diagnostics import run_firmware_frame
        for row in run_firmware_frame(ROOT / config['dataset'], OUTPUTS):
            print(f"T0.1 {row['method']} h={row['height_difference_m']}: MAE={row['position_mae_m']:.4f} m, P90={row['position_p90_m']:.4f} m")
    if args.t0_mlp_check:
        from workflow.diagnostics import run_mlp_check
        for row in run_mlp_check(ROOT / config['dataset'], OUTPUTS):
            print(f"T0.2 {row['fold_id']} {row['method']}: train MAE={row['train_mae_deg']:.2f} deg, evaluation MAE={row['evaluation_mae_deg']:.2f} deg")
    if args.t0_bearing_label:
        from workflow.diagnostics import run_bearing_label_check
        for row in run_bearing_label_check(OUTPUTS):
            print(f"T0.3 h={row['height_difference_m']} {row['method']}: MAE={row['mae_deg']:.2f} deg, trig_bearing-minus-method={row['trig_bearing_minus_method_mae_deg']:+.2f} deg")
    if args.phase1 or args.t1_characteristic:
        from workflow.diagnostics import run_phase_characteristic
        for row in run_phase_characteristic(ROOT / config['dataset'], OUTPUTS):
            if 'scale' in row:
                print(f"T1.1 h={row['height_difference_m']}: scale={row['scale']:.4f} [{row['scale_ci_low']:.4f}, {row['scale_ci_high']:.4f}], "
                      f"offset={row['offset_deg']:.2f} [{row['offset_ci_low_deg']:.2f}, {row['offset_ci_high_deg']:.2f}] deg")
            else:
                print(f"T1.1 {row['scope']} {row['method']}: MAE={row['mae_deg']:.2f} [{row['mae_ci_low_deg']:.2f}, {row['mae_ci_high_deg']:.2f}] deg")
    if args.phase1 or args.t1_invertibility:
        from workflow.diagnostics import run_invertibility
        for row in run_invertibility(ROOT / config['dataset'], OUTPUTS):
            print(f"T1.2 {row['tag']} boresight={row['boresight_deg']:+.0f}: physical MAE={row['physical_inverse_mae_deg']:.2f} deg, "
                  f"trig MAE={row['trigonometric_global_mae_deg']:.2f} deg, useful={row['useful']}")
    if args.phase1 or args.t1_elevation:
        from workflow.diagnostics import run_elevation
        for row in run_elevation(OUTPUTS):
            print(f"T1.3 {row['method']} elev={row['elevation_deg']:.1f}: h4(2d)-h2(d)={row['h4_2d_minus_h2_d_deg']:+.2f} "
                  f"[{row['h4_2d_minus_h2_d_ci_low_deg']:+.2f}, {row['h4_2d_minus_h2_d_ci_high_deg']:+.2f}] deg")
    if args.phase1 or args.t1_rotation:
        from workflow.diagnostics import run_rotation
        for row in run_rotation(ROOT / config['dataset'], OUTPUTS):
            if row['height_difference_m'] == 'all':
                print(f"T1.4 {row['tag']}: rotation spread={row['max_rotation_spread_deg']:.1f} deg, explained={row['rotation_explained_share']:.2f}")
    if args.phase1 or args.t1_session:
        from workflow.diagnostics import run_session
        for row in run_session(ROOT / config['dataset'], OUTPUTS):
            if row['excludes_zero'] or row['tag'] == 'all':
                print(f"T1.5 {row['quantity']} {row['tag']} {row['comparison']}: {row['difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
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
    if args.phase2:
        args.loho_distance, args.pdoa_gate = True, True
        args.pdoa_elevation, args.position_elevation = ["lodo", "loho", "lobo"], ["lodo", "loho"]
    if args.compare_distance or args.lodo_distance or args.loho_distance:
        from workflow.distance_benchmark import run_benchmark
        for enabled, protocol in ((args.compare_distance, 'grouped'), (args.lodo_distance, 'lodo'),
                                  (args.loho_distance, 'loho')):
            if enabled:
                results = run_benchmark(ROOT / config['dataset'], OUTPUTS,
                                        ANALYSIS / 'configs/distance_models.json', protocol)
                for row in results:
                    print(f"{protocol} {row['method']}: MAE={row['mae_m']:.6f} m, RMSE={row['rmse_m']:.6f} m, P90={row['p90_absolute_error_m']:.6f} m")

    for protocol in args.pdoa_elevation or ():
        from workflow.phase_elevation import run_phase_elevation
        for row in run_phase_elevation(ROOT / config['dataset'], OUTPUTS, protocol):
            print(f"T2.2 {protocol} {row['stratum']} {row['method']}: MAE={row['mae_deg']:.2f} [{row['mae_ci_low_deg']:.2f}, {row['mae_ci_high_deg']:.2f}] deg, "
                  f"vs trig={row['minus_trigonometric_global_mae_deg']:+.2f} [{row['minus_trigonometric_global_ci_low_deg']:+.2f}, {row['minus_trigonometric_global_ci_high_deg']:+.2f}]")
    if args.pdoa_gate:
        from workflow.phase_elevation import run_gate
        for row in run_gate(OUTPUTS):
            print(f"T2.3 coverage>={row['coverage_target']:.0%}: |pdoa|<={row['phase_limit_deg']:g}, range<={row['range_limit_m']:g} m, "
                  f"elev<={row['elevation_limit_deg']:g} -> coverage={row['coverage']:.3f}, MAE={row['elevation_range_mae_deg']:.2f} deg "
                  f"[{row['elevation_range_mae_ci_low_deg']:.2f}, {row['elevation_range_mae_ci_high_deg']:.2f}], trig MAE={row['trigonometric_global_mae_deg']:.2f}")
    for protocol in args.position_elevation or ():
        from workflow.phase_elevation import run_position
        for row in run_position(OUTPUTS, protocol):
            if row['height_difference_m'] == 'all':
                print(f"T2.4 {protocol} {row['scope']} {row['method']}: coverage={row['coverage']:.3f}, MAE={row['position_mae_m']:.3f} m "
                      f"[{row['position_mae_ci_low_m']:.3f}, {row['position_mae_ci_high_m']:.3f}], P90={row['position_p90_m']:.3f} m")

    if args.final_test:
        from workflow.final_test import run_final_test
        result = run_final_test(ROOT / config['dataset'], OUTPUTS, force=args.force_final_test)
        for row in result['range']:
            if row['height_difference_m'] == 'all':
                print(f"T3.2 range {row['rows']} {row['method']}: MAE={row['mae_m']:.4f} m [{row['mae_ci_low_m']:.4f}, {row['mae_ci_high_m']:.4f}]")
        for row in result['pdoa']:
            if row['height_difference_m'] == 'all':
                print(f"T3.2 pdoa {row['scope']} {row['method']}: coverage={row['coverage']:.3f}, MAE={row['mae_deg']:.2f} deg [{row['mae_ci_low_deg']:.2f}, {row['mae_ci_high_deg']:.2f}]")
        for row in result['position']:
            if row['height_difference_m'] == 'all':
                print(f"T3.2 position {row['scope']} {row['method']}: coverage={row['coverage']:.3f}, MAE={row['position_mae_m']:.3f} m "
                      f"[{row['position_mae_ci_low_m']:.3f}, {row['position_mae_ci_high_m']:.3f}], P90={row['position_p90_m']:.3f} m")


if __name__ == "__main__":
    main()
