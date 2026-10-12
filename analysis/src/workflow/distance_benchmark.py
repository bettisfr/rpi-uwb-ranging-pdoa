"""Grouped validation and leave-one-distance-out on the development pool."""

import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .distance_models import METHODS, make_model
from .offset import distance_quality_counts, load_assigned_rows


def make_folds(distance, subset, protocol, heights=None):
    """Original test is excluded from both training and evaluation in all folds."""
    development = np.isin(subset, ['train', 'validation'])
    if protocol == 'grouped':
        yield 'grouped_validation', subset == 'train', subset == 'validation'
    elif protocol == 'lodo':
        for held in sorted(np.unique(distance[development])):
            train = development & (distance != held)
            evaluate = development & (distance == held)
            if not train.any() or not evaluate.any():
                raise ValueError('Empty leave-one-distance-out partition')
            yield f'distance_{held:g}m', train, evaluate
    elif protocol == 'loho':
        for held in sorted(np.unique(heights[development])):
            train = development & (heights != held)
            evaluate = development & (heights == held)
            if not train.any() or not evaluate.any():
                raise ValueError('Empty leave-one-height-out partition')
            yield f'height_{held:g}m', train, evaluate
    else:
        raise ValueError(f'Unknown protocol: {protocol}')


def metrics(y, prediction):
    error = prediction - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(samples=len(y), bias_m=float(error.mean()), mae_m=float(np.abs(error).mean()),
                rmse_m=float(np.sqrt(np.mean(error**2))), std_m=float(error.std()),
                distance_r2=1-float(np.sum(error**2))/sst if sst else None,
                p90_absolute_error_m=float(np.quantile(np.abs(error), .9, method='inverted_cdf')),
                max_absolute_error_m=float(np.abs(error).max()))


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(dataset: Path, output: Path, config_path: Path, protocol='grouped'):
    config = json.loads(config_path.read_text())
    rows, split_manifest, digest = load_assigned_rows(dataset, output)
    # Keep source-row identities through eligibility filtering.
    eligible = [r for r in rows if r['quality_enabled']]
    excluded = {s: sum(r['subset'] == s and not r['quality_enabled'] for r in rows)
                for s in ('train', 'validation', 'test')}
    X = np.array([float(r['estimated_ground_distance_m']) for r in eligible]).reshape(-1, 1)
    y = np.array([float(r['target_ground_distance_m']) for r in eligible])
    subset = np.array([r['subset'] for r in eligible])
    heights = np.array([float(r['height_difference_m']) for r in eligible])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite model input or target')
    destination = output / {'grouped': 'distance_comparison', 'lodo': 'distance_lodo', 'loho': 'distance_loho'}[protocol]
    destination.mkdir(exist_ok=True)
    manifest = dict(status='running', protocol=protocol, config=config,
                    source_sha256=digest, split=split_manifest, sklearn_version=sklearn.__version__,
                    numpy_version=np.__version__, input='estimated_ground_distance_m',
                    target='target_ground_distance_m', fit_weighting='per sample',
                    outlier_policy=('Source rows retained; disable valid horizontal readings whose absolute '
                                    'known-geometry error exceeds 3 m, plus undefined projections.'),
                    excluded_by_subset=excluded,
                    quality_counts_all_rows=distance_quality_counts(rows),
                    test_policy='Original grouped test is excluded from all fits and evaluations',
                    fold_scope={'grouped': 'Original train fits, original validation evaluates',
                                'lodo': 'Train+validation development pool; all tags, rotations and heights of held distance are excluded from fitting',
                                'loho': 'Train+validation development pool; all tags, rotations and distances of held height are excluded from fitting'}[protocol],
                    parameter_selection='Fixed before validation/LODO, no tuning on evaluation rows',
                    p90_definition='nearest rank', fold_models=[])
    manifest_path = destination / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    summary, diagnostics, pooled = [], [], {name: [] for name in ('raw', *METHODS)}
    assignment_path = destination / 'assignments.csv'
    with assignment_path.open('w', newline='') as stream:
        csv.writer(stream).writerow(('fold_id', 'csv_line', 'role'))
    for fold, train, evaluate in make_folds(y, subset, protocol, heights):
        if not train.any() or not evaluate.any():
            raise ValueError('Empty training/evaluation subset')
        with assignment_path.open('a', newline='') as stream:
            writer = csv.writer(stream)
            for i in np.flatnonzero(train | evaluate):
                writer.writerow((fold, eligible[i]['csv_line'], 'train' if train[i] else 'evaluate'))
        fold_path = destination / fold
        fold_path.mkdir(exist_ok=True)
        indices = np.flatnonzero(evaluate)
        estimates = {'raw': X[evaluate, 0]}
        for name in METHODS:
            print(f'{protocol} {fold}: fitting {name} ({train.sum()} samples)', flush=True)
            start = time.monotonic()
            model = make_model(name, config)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(X[train], y[train])
                estimates[name] = model.predict(X[evaluate])
            joblib.dump(model, fold_path / f'{name}.joblib')
            info = dict(fold_id=fold, method=name, training_samples=int(train.sum()),
                        evaluated_samples=int(evaluate.sum()), elapsed_s=time.monotonic()-start,
                        warnings=[str(w.message) for w in caught])
            if name == 'svr':
                info['unique_training_pairs'] = model.unique_pairs_
            if name == 'mlp':
                network = model.regressor_.named_steps['mlpregressor']
                info.update(iterations=int(network.n_iter_), final_training_loss=float(network.loss_))
            manifest['fold_models'].append(info)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        # Wide CSV: one row per evaluation sample, all methods on that same row.
        prediction_rows = []
        for pos, i in enumerate(indices):
            prediction_rows.append(dict(csv_line=eligible[i]['csv_line'],
                                        group_id=eligible[i]['group_id'], tag=eligible[i]['tag'],
                                        height_difference_m=heights[i], target_ground_distance_m=y[i],
                                        **{f'{name}_m': float(values[pos]) for name, values in estimates.items()}))
        save_csv(fold_path / 'predictions.csv', prediction_rows)
        for name, prediction in estimates.items():
            pooled[name].append((y[evaluate], prediction))
            for h in [None, *sorted(np.unique(heights[evaluate]))]:
                select = np.ones(len(indices), dtype=bool) if h is None else heights[evaluate] == h
                summary.append(dict(fold_id=fold, method=name, height_difference_m='all' if h is None else h,
                                    **metrics(y[evaluate][select], prediction[select])))
            for d in sorted(np.unique(y[evaluate])):
                select = y[evaluate] == d
                diagnostics.append(dict(fold_id=fold, method=name, distance_m=d,
                                        **metrics(y[evaluate][select], prediction[select])))
        save_csv(destination / 'metrics.csv', summary)
        save_csv(destination / 'metrics_by_distance.csv', diagnostics)
        print(f'{protocol} {fold} complete', flush=True)
    overall = [dict(method=name, **metrics(np.concatenate([v[0] for v in pairs]),
                                         np.concatenate([v[1] for v in pairs])))
               for name, pairs in pooled.items()]
    save_csv(destination / 'overall_metrics.csv', overall)
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return overall
