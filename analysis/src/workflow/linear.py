"""OLS horizontal-distance calibration and validation-only model comparison."""

import csv
import json
import math
from collections import defaultdict
from statistics import fmean

from .offset import (distance_metrics, distance_quality_counts, fit_offset,
                     is_distance_enabled, load_assigned_rows)


def fit_linear(rows):
    pairs = [(float(r['estimated_ground_distance_m']), float(r['target_ground_distance_m']))
             for r in rows if r['subset'] == 'train' and is_distance_enabled(r)]
    if len(pairs) < 2 or not all(math.isfinite(v) for pair in pairs for v in pair):
        raise ValueError('Need finite training observations')
    xmean = fmean(x for x, _ in pairs)
    ymean = fmean(y for _, y in pairs)
    denominator = sum((x - xmean) ** 2 for x, _ in pairs)
    if denominator == 0:
        raise ValueError('Training input distance has zero variance')
    a = sum((x - xmean) * (y - ymean) for x, y in pairs) / denominator
    return a, ymean - a * xmean


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No results to export to {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_linear(dataset, output, *, include_quadratic=False, include_forest=False):
    rows, manifest, digest = load_assigned_rows(dataset, output)
    # Test observations are not evaluated during development/model selection.
    rows = [r for r in rows if r['subset'] in ('train', 'validation')]
    a, b = fit_linear(rows)
    offset = fit_offset(rows)
    forest = None
    forest_predictions = {}
    if include_forest:
        from .forest import fit_forest
        forest = fit_forest(rows)
        eligible = [r for r in rows if is_distance_enabled(r)]
        values = forest.predict([[float(r['estimated_ground_distance_m'])] for r in eligible])
        forest_predictions = {r['csv_line']: float(value) for r, value in zip(eligible, values)}
    quadratic = None
    if include_quadratic:
        from .quadratic import fit_quadratic, predict_quadratic
        quadratic = fit_quadratic(rows)
    predictions = []
    grouped = defaultdict(list)
    detail = defaultdict(list)
    excluded = defaultdict(int)
    for r in rows:
        if not is_distance_enabled(r):
            excluded[r['subset']] += 1
            continue
        truth = float(r['target_ground_distance_m'])
        raw = float(r['estimated_ground_distance_m'])
        estimates = [('raw', raw), ('constant_offset', raw - offset),
                     ('linear_regression', a * raw + b)]
        if quadratic is not None:
            estimates.append(('quadratic_regression', predict_quadratic(raw, quadratic)))
        if forest is not None:
            estimates.append(('random_forest', forest_predictions[r['csv_line']]))
        for method, estimate in estimates:
            record = {k: r[k] for k in ('csv_line', 'experiment_id', 'run_file', 'tag',
                       'group_id', 'subset', 'height_difference_m', 'tag_rotation_deg')}
            record.update(method=method, target_ground_distance_m=truth,
                          raw_ground_distance_m=raw, prediction_m=estimate, error_m=estimate - truth)
            predictions.append(record)
            for height in ('all', r['height_difference_m']):
                grouped[(r['subset'], height, method)].append((truth, estimate))
            if r['subset'] == 'validation':
                detail[(r['height_difference_m'], truth, r['tag'], method)].append((truth, estimate))

    def metrics(values):
        truth, estimate = map(list, zip(*values))
        result = distance_metrics(truth, estimate)
        absolute = sorted(abs(p - t) for t, p in values)
        # Nearest-rank quantiles, documented in the manifest.
        result.update(p90_absolute_error_m=absolute[math.ceil(.9 * len(absolute)) - 1],
                      max_absolute_error_m=absolute[-1])
        return result

    summary = [dict(subset=s, height_difference_m=h, method=m, **metrics(values))
               for (s, h, m), values in sorted(grouped.items())]
    diagnostics = [dict(subset='validation', height_difference_m=h,
                        target_ground_distance_m=d, tag=t, method=m, **metrics(values))
                   for (h, d, t, m), values in sorted(detail.items())]
    destination = output / ('random_forest' if include_forest else 'quadratic_regression' if include_quadratic else 'linear_regression')
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'predictions.csv', predictions)
    write_csv(destination / 'metrics.csv', summary)
    write_csv(destination / 'validation_by_configuration.csv', diagnostics)
    model = dict(model='linear_regression', slope=a, intercept_m=b,
                 offset_baseline_m=offset, formula='rho_corrected = slope * rho_measured + intercept_m',
                 input='estimated_ground_distance_m', target='target_ground_distance_m',
                 fit_subset='train', evaluated_subsets=['train', 'validation'],
                 fit_scope='all heights pooled', weighting='per sample',
                 training_samples=sum(r['subset'] == 'train' and is_distance_enabled(r) for r in rows),
                 source_sha256=digest, split=manifest, excluded_by_subset=dict(excluded),
                 diagnostics='Validation grouped by height/distance/tag, pooling available rotations; p90 uses nearest rank; std ddof=0; distance R2 null for constant truth')
    model['outlier_policy'] = 'Source rows are retained; undefined projections and gross known-geometry errors are disabled.'
    model['quality_counts_all_rows'] = distance_quality_counts(rows)
    if quadratic is not None:
        model.update(model='quadratic_regression', linear_baseline=dict(slope=a, intercept_m=b),
                     quadratic=quadratic, formula='z=(rho_measured-center_m)/scale_m; rho_corrected=c0+c1*z+c2*z*z')
        del model['slope'], model['intercept_m']
    if forest is not None:
        import joblib
        import sklearn
        model.update(model='random_forest', parameters=forest.get_params(),
                     sklearn_version=sklearn.__version__,
                     feature_names=['estimated_ground_distance_m'],
                     formula='RandomForestRegressor.predict([[rho_measured]])',
                     parameter_selection='Fixed in advance; no validation tuning',
                     artifact='model.joblib',
                     limitation='Grouped configurations share nominal distance levels across subsets; this does not establish generalization to unseen distances.')
        joblib.dump(forest, destination / 'model.joblib')
    (destination / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    return model, summary
