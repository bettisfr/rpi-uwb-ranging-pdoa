"""Grouped-test position comparison with fixed quadratic/offset and harmonic models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from workflow.distance_models import make_model
from workflow.offset import load_assigned_rows
from workflow.phase import wrap
from workflow.phase_trigonometric import fit_trigonometric, predict_correction
from workflow.position_benchmark import metrics, reconstruct, write_csv


ROOT = Path(__file__).resolve().parents[2]


def confidence_interval(errors, groups):
    """95% grouped bootstrap interval: 1000 replicates, seed 42, sample weighting."""
    _, indices = np.unique(groups, return_inverse=True)
    counts = np.bincount(indices).astype(float)
    sums = np.bincount(indices, weights=errors)
    weights = np.random.default_rng(42).multinomial(
        len(counts), np.full(len(counts), 1. / len(counts)), size=1000)
    means = (weights @ sums) / (weights @ counts)
    return np.quantile(means, [.025, .975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'analysis/csv/static_measurements.csv')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'analysis/outputs')
    args = parser.parse_args()
    rows, split, digest = load_assigned_rows(args.dataset, args.outputs)
    dev = [row for row in rows if row['subset'] in ('train', 'validation')]
    train = [row for row in dev if row['quality_enabled']]
    test = [row for row in rows if row['subset'] == 'test' and row['quality_enabled']]
    column = lambda records, key: np.asarray([float(row[key]) for row in records])
    phase = column(test, 'pdoa_deg')
    dev_phase = column(dev, 'pdoa_deg')
    phase_model = fit_trigonometric(dev_phase, column(dev, 'eta_expected_deg'),
                                    np.full(len(dev), 'train', dtype=object))
    correction, _, _ = predict_correction(phase, phase_model)
    corrected_phase = wrap(phase - correction)
    height = column(test, 'height_difference_m')
    measured_range = column(test, 'range_m')
    horizontal = column(test, 'estimated_ground_distance_m')
    elevation = np.rad2deg(np.arctan2(height, horizontal))
    selected = (measured_range <= 25.) & (elevation <= 45.)
    truth_x, truth_y = column(test, 'gt_x_m'), column(test, 'gt_y_m')
    groups = np.asarray([f"{row['tag']}|{float(row['height_difference_m']):g}|"
                         f"{float(row['target_ground_distance_m']):g}|{float(row['tag_rotation_deg']):g}"
                         for row in test])
    config = json.loads((ROOT / 'analysis/configs/distance_models.json').read_text())
    destination = args.outputs / 'position_quadratic_test'
    destination.mkdir(parents=True, exist_ok=True)
    results, estimates = [], {}
    for name in ('constant_offset', 'quadratic_regression'):
        model = make_model(name, config)
        model.fit(column(train, 'estimated_ground_distance_m').reshape(-1, 1),
                  column(train, 'target_ground_distance_m'))
        rho = model.predict(horizontal.reshape(-1, 1))
        x, y, clipped = reconstruct(rho, height, corrected_phase)
        errors = np.hypot(x - truth_x, y - truth_y)
        estimates[name] = (rho, x, y, errors, clipped)
        joblib.dump(model, destination / f'{name}.joblib')
        for scope, mask in (('all', np.ones(len(test), dtype=bool)), ('selected', selected)):
            for h in ('all', *sorted(set(height))):
                stratum = np.ones(len(test), dtype=bool) if h == 'all' else height == h
                use = mask & stratum
                low, high = confidence_interval(errors[use], groups[use])
                results.append(dict(method=f'{name}_global_harmonic', scope=scope,
                    height_difference_m=h, coverage=float(use.sum() / stratum.sum()),
                    **metrics(truth_x[use], truth_y[use], x[use], y[use], clipped[use]),
                    position_mae_ci_low_m=float(low), position_mae_ci_high_m=float(high)))
    write_csv(destination / 'metrics.csv', results)
    write_csv(destination / 'predictions.csv', [dict(csv_line=row['csv_line'], tag=row['tag'],
        height_difference_m=height[i], true_x_m=truth_x[i], true_y_m=truth_y[i],
        corrected_phase_deg=corrected_phase[i], selected=bool(selected[i]),
        **{f'{name}_{field}': float(values[j][i]) for name, values in estimates.items()
           for j, field in enumerate(('rho_m', 'x_m', 'y_m', 'position_error_m', 'clipped'))})
        for i, row in enumerate(test)])
    manifest = dict(source_sha256=digest, split=split, fit_pool='train + validation',
        evaluated='original grouped test; distance-quality-enabled rows',
        selection=dict(measured_range_max_m=25., estimated_elevation_max_deg=45.),
        policy='Fixed model forms and thresholds; no tuning on test',
        phase_model=phase_model, range_training_samples=len(train), phase_training_samples=len(dev),
        test_samples=len(test), bootstrap=dict(replicates=1000, seed=42,
        unit='tag, height, true horizontal distance, rotation'))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for result in results:
        if result['height_difference_m'] == 'all':
            print(f"{result['method']} {result['scope']}: n={result['samples']}, "
                  f"coverage={100*result['coverage']:.2f}%, MAE={result['position_mae_m']:.6f} m, "
                  f"CI=[{result['position_mae_ci_low_m']:.6f}, {result['position_mae_ci_high_m']:.6f}], "
                  f"RMSE={result['position_rmse_m']:.6f} m, P90={result['position_p90_m']:.6f} m")
    print(f'Outputs: {destination}')


if __name__ == '__main__':
    main()
