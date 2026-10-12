"""T3.2: one-time evaluation of the frozen models on the grouped test subset.

The models and gate are pre-registered in analysis/README.md ("Phase 3").
Everything is fitted on train + validation; the test subset is touched once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bootstrap import mean_interval
from .diagnostics import _static_arrays, fit_phase_characteristic, invert_characteristic
from .phase import circular_summary, wrap
from .phase_elevation import estimated_elevation_deg, gate_mask
from .phase_trigonometric import fit_trigonometric, predict_correction
from .position_benchmark import metrics as position_metrics, reconstruct, write_csv


GATE_CELL = (180., 25., 45.)  # |pdoa| deg, measured range m, estimated elevation deg


def _interval(values, groups):
    estimate, (low, high), _ = mean_interval(values, groups)
    return dict(ci_low=low, ci_high=high)


def run_final_test(dataset: Path, output: Path, force=False) -> dict:
    destination = output / "final_test"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete" and not force:
        raise SystemExit("Final test already evaluated; results are frozen in outputs/final_test/.")
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    dev, split, digest = _static_arrays(dataset, output)
    test, _, _ = _static_arrays(dataset, output, subsets=("test",))
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", test="T3.2", source_sha256=digest, split=split,
                    fit_pool="train + validation", evaluated="grouped test", forced_rerun=force,
                    previous_status=previous.get("status") if previous else None,
                    gate=dict(phase_limit_deg=GATE_CELL[0], range_limit_m=GATE_CELL[1],
                              elevation_limit_deg=GATE_CELL[2]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Frozen models, fitted once on the development pool.
    enabled = dev["quality"]
    offset = float(np.mean(dev["ground_hat"][enabled] - dev["distance"][enabled]))
    physical = fit_phase_characteristic(dev["phase"], dev["expected"], dev["group"], n_boot=0)
    trig = fit_trigonometric(dev["phase"], dev["expected"], np.full(len(dev["phase"]), "train", dtype=object))
    manifest["frozen"] = dict(range_offset_m=offset, physical_scale=physical["scale"],
                              physical_offset_deg=physical["offset_deg"], trigonometric=trig)

    groups = test["group"]
    elevation_hat = estimated_elevation_deg(test["range_m"], test["height"])
    gated = gate_mask(test["phase"], test["range_m"], elevation_hat, GATE_CELL)

    # Range on test.
    range_rows = []
    defined = np.isfinite(test["ground_hat"])
    for mask_name, mask in (("quality_enabled", test["quality"]), ("all_defined", defined)):
        for method, estimate in (("raw", test["ground_hat"]), ("constant_offset", test["ground_hat"] - offset)):
            for h in [None, *sorted(set(test["height"]))]:
                select = mask & (True if h is None else test["height"] == h)
                error = estimate[select] - test["distance"][select]
                range_rows.append(dict(rows=mask_name, method=method, height_difference_m="all" if h is None else h,
                                       samples=int(select.sum()), bias_m=float(error.mean()),
                                       mae_m=float(np.abs(error).mean()),
                                       rmse_m=float(np.sqrt(np.mean(error ** 2))),
                                       p90_absolute_error_m=float(np.quantile(np.abs(error), .9, method="inverted_cdf")),
                                       **{f"mae_{k}_m": v for k, v in _interval(np.abs(error), groups[select]).items()}))

    # PDoA on test.
    correction, _, _ = predict_correction(test["phase"], trig)
    phases = {"raw": wrap(test["phase"]), "trigonometric_global": wrap(test["phase"] - correction),
              "physical_inverse": invert_characteristic(test["phase"], physical)}
    phase_rows = []
    for method, estimate in phases.items():
        error = wrap(estimate - test["expected"])
        for scope, mask in (("all", np.ones(len(error), dtype=bool)), ("gated", gated)):
            for h in [None, *sorted(set(test["height"]))]:
                select = mask & (True if h is None else test["height"] == h)
                if select.any():
                    stratum = np.ones(len(error), dtype=bool) if h is None else test["height"] == h
                    phase_rows.append(dict(method=method, scope=scope, height_difference_m="all" if h is None else h,
                                           coverage=float(select.sum() / stratum.sum()),
                                           **circular_summary(error[select]),
                                           **{f"mae_{k}_deg": v for k, v in _interval(np.abs(error[select]), groups[select]).items()}))

    # Position on test (quality-enabled rows, as in T2.4).
    usable = test["quality"]
    rho = test["ground_hat"][usable] - offset
    x_true, y_true, height = test["gt_x"][usable], test["gt_y"][usable], test["height"][usable]
    estimates = {"firmware_raw_xy": (test["firmware_x"][usable], test["firmware_y"][usable], np.zeros(usable.sum(), dtype=bool))}
    for method in ("raw", "trigonometric_global", "physical_inverse"):
        estimates[f"geometry_offset_{method}"] = reconstruct(rho, height, phases[method][usable])
    position_rows = []
    for method, (x_hat, y_hat, clipped) in estimates.items():
        error = np.hypot(x_hat - x_true, y_hat - y_true)
        for scope, mask in (("all", np.ones(len(error), dtype=bool)), ("gated", gated[usable])):
            for h in [None, *sorted(set(height))]:
                select = mask & (True if h is None else height == h)
                stratum = np.ones(len(error), dtype=bool) if h is None else height == h
                position_rows.append(dict(method=method, scope=scope, height_difference_m="all" if h is None else h,
                                          coverage=float(select.sum() / stratum.sum()),
                                          **position_metrics(x_true[select], y_true[select], x_hat[select],
                                                             y_hat[select], clipped[select]),
                                          **{f"position_mae_{k}_m": v for k, v in
                                             _interval(error[select], groups[usable][select]).items()}))

    write_csv(destination / "range_metrics.csv", range_rows)
    write_csv(destination / "pdoa_metrics.csv", phase_rows)
    write_csv(destination / "position_metrics.csv", position_rows)
    manifest.update(status="complete", test_samples=int(len(test["phase"])),
                    test_quality_enabled=int(test["quality"].sum()), test_gated_share=float(gated.mean()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(range=range_rows, pdoa=phase_rows, position=position_rows)
