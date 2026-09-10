"""Training-only constant correction of the projected horizontal distance."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from statistics import fmean, pstdev
from pathlib import Path


# This campaign has surveyed reference distances.  Retain source rows but do
# not use isolated readings that disagree with the known horizontal geometry by
# more than this deliberately conservative threshold.
DISTANCE_QUALITY_THRESHOLD_M = 3.0


def is_distance_enabled(row):
    """Return whether a row is usable by a distance-calibration model.

    Undefined horizontal projections remain in the source table but cannot be
    a target.  The remaining disabled rows are gross glitches identified
    against the controlled test-bed geometry; they are never deleted.
    """
    value = row['ground_distance_error_m']
    return bool(value) and math.isfinite(float(value)) and abs(float(value)) <= DISTANCE_QUALITY_THRESHOLD_M


def distance_quality_counts(rows):
    """Count invalid projections and retained-but-disabled gross glitches."""
    counts = {'invalid_projection': 0, 'disabled_gross_error': 0}
    for row in rows:
        if row['ground_distance_error_m'] == '':
            counts['invalid_projection'] += 1
        elif not is_distance_enabled(row):
            counts['disabled_gross_error'] += 1
    return counts


def fit_offset(rows):
    """Sample-weighted mean signed error, fitted exclusively on training rows."""
    errors = [float(row['ground_distance_error_m']) for row in rows
              if row['subset'] == 'train' and is_distance_enabled(row)]
    if not errors or not all(math.isfinite(value) for value in errors):
        raise ValueError('Training needs finite eligible distance errors')
    return fmean(errors)


def distance_metrics(truth, prediction):
    """Distance accuracy in meters; R2 concerns distance, not error prediction."""
    if len(truth) != len(prediction) or not truth:
        raise ValueError('Equal nonempty truth/prediction vectors required')
    errors = [p - t for t, p in zip(truth, prediction)]
    sse = sum(e * e for e in errors)
    mean = fmean(truth)
    sst = sum((t - mean) ** 2 for t in truth)
    return dict(samples=len(errors), bias_m=fmean(errors), mae_m=fmean(map(abs, errors)),
                rmse_m=math.sqrt(sse / len(errors)), std_m=pstdev(errors),
                distance_r2=1 - sse / sst if sst else None)


def load_assigned_rows(dataset: Path, output: Path):
    """Load observations joined to the existing, hash-checked split."""
    manifest = json.loads((output / 'static_grouped_manifest.json').read_text())
    with dataset.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != manifest['source_sha256']:
        raise ValueError('Dataset differs from the split manifest; rebuild splits first')
    keys = ('experiment_id', 'run_file', 'tag')
    with (output / 'static_grouped_assignments.csv').open(newline='') as source:
        assignments = {}
        for row in csv.DictReader(source):
            key = tuple(row[k] for k in keys)
            if key in assignments:
                raise ValueError('Duplicate assignment key')
            assignments[key] = row
    with dataset.open(newline='') as source:
        rows = []
        for line, row in enumerate(csv.DictReader(source), 2):
            assignment = assignments[tuple(row[k] for k in keys)]
            rows.append({**row, 'subset': assignment['subset'],
                         'group_id': assignment['group_id'], 'csv_line': line,
                         'quality_enabled': is_distance_enabled(row)})
    return rows, manifest, digest


def run_offset(dataset: Path, output: Path):
    """Consume verified grouped assignments and export a frozen baseline."""
    rows, manifest, digest = load_assigned_rows(dataset, output)
    keys = ('experiment_id', 'run_file', 'tag')
    offset = fit_offset(rows)
    destination = output / 'constant_offset'
    destination.mkdir(exist_ok=True)
    groups = defaultdict(list)
    excluded = defaultdict(int)
    fields = ('csv_line', *keys, 'group_id', 'subset', 'height_difference_m',
              'target_ground_distance_m', 'raw_ground_distance_m',
              'corrected_ground_distance_m', 'raw_error_m', 'corrected_error_m')
    with (destination / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row['quality_enabled']:
                excluded[row['subset']] += 1
                continue
            truth = float(row['target_ground_distance_m'])
            raw = float(row['estimated_ground_distance_m'])
            corrected = raw - offset
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             'raw_ground_distance_m': raw,
                             'corrected_ground_distance_m': corrected,
                             'raw_error_m': raw - truth,
                             'corrected_error_m': corrected - truth})
            for height in ('all', row['height_difference_m']):
                groups[(row['subset'], height)].append((truth, raw, corrected))
    results = []
    for (subset, height), values in sorted(groups.items()):
        truth, raw, corrected = map(list, zip(*values))
        for method, predictions in (('raw', raw), ('constant_offset', corrected)):
            results.append(dict(subset=subset, height_difference_m=height, method=method,
                                **distance_metrics(truth, predictions)))
    with (destination / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    model = dict(model='constant_offset', target='ground_distance_error_m',
                 offset_m=offset, correction='corrected_horizontal_distance = raw_horizontal_distance - offset_m',
                 fit_subset='train', fit_scope='all three heights pooled', weighting='per sample',
                 training_samples=sum(r['subset'] == 'train' and r['quality_enabled'] for r in rows),
                 excluded_by_subset=dict(excluded), source_sha256=digest,
                 split=manifest, metric_note='std uses ddof=0; R2 evaluates horizontal distance; null if truth is constant')
    model['outlier_policy'] = ('Keep every source row; disable only valid horizontal observations with '
                               f'absolute known-geometry error above {DISTANCE_QUALITY_THRESHOLD_M:g} m')
    model['quality_counts_all_rows'] = distance_quality_counts(rows)
    (destination / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    return model, results
