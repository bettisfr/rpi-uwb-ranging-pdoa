"""Evaluate fixed parametric range models on the original grouped test split."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from workflow.distance_benchmark import metrics
from workflow.distance_models import make_model
from workflow.offset import load_assigned_rows


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'analysis/csv/static_measurements.csv')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'analysis/outputs')
    parser.add_argument('--include-forest', action='store_true', help='Also evaluate the fixed Random Forest baseline')
    args = parser.parse_args()
    rows, split, digest = load_assigned_rows(args.dataset, args.outputs)
    eligible = [row for row in rows if row['quality_enabled']]
    x = np.array([float(row['estimated_ground_distance_m']) for row in eligible]).reshape(-1, 1)
    y = np.array([float(row['target_ground_distance_m']) for row in eligible])
    subsets = np.array([row['subset'] for row in eligible])
    heights = np.array([float(row['height_difference_m']) for row in eligible])
    train = np.isin(subsets, ['train', 'validation'])
    test = subsets == 'test'
    config = json.loads((ROOT / 'analysis/configs/distance_models.json').read_text())
    destination = args.outputs / 'range_parametric_test'
    destination.mkdir(parents=True, exist_ok=True)
    predictions = {'raw': x[test, 0]}
    methods = ['constant_offset', 'linear_regression', 'quadratic_regression']
    if args.include_forest:
        methods.append('random_forest')
    for name in methods:
        model = make_model(name, config)
        model.fit(x[train], y[train])
        predictions[name] = model.predict(x[test])
        joblib.dump(model, destination / f'{name}.joblib')
    results = []
    for name, prediction in predictions.items():
        for height in ('all', *sorted(np.unique(heights[test]))):
            select = np.ones(test.sum(), dtype=bool) if height == 'all' else heights[test] == height
            results.append(dict(method=name, height_difference_m=height,
                                **metrics(y[test][select], prediction[select])))
    with (destination / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (destination / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['csv_line', 'height_difference_m', 'target_ground_distance_m', *predictions])
        for position, index in enumerate(np.flatnonzero(test)):
            writer.writerow([eligible[index]['csv_line'], heights[index], y[index],
                             *[values[position] for values in predictions.values()]])
    manifest = dict(source_sha256=digest, split=split, fit_pool='train + validation',
                    evaluated='grouped test', training_samples=int(train.sum()),
                    test_samples=int(test.sum()), input='estimated_ground_distance_m',
                    target='target_ground_distance_m', models=list(predictions),
                    policy='Fixed model forms; no tuning or model selection on test; existing quality filter applied.')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for result in results:
        if result['height_difference_m'] == 'all':
            print(f"{result['method']}: n={result['samples']}, MAE={100*result['mae_m']:.4f} cm, "
                  f"RMSE={100*result['rmse_m']:.4f} cm")
    print(f'Outputs: {destination}')


if __name__ == '__main__':
    main()
