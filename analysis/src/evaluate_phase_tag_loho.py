"""LOHO diagnostics: tag-wise harmonic correction and label-only constant."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from workflow.offset import load_assigned_rows
from workflow.phase import circular_summary, wrap
from workflow.phase_trigonometric import fit_trigonometric, predict_correction


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'analysis/csv/static_measurements.csv')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'analysis/outputs')
    args = parser.parse_args()
    rows, split, digest = load_assigned_rows(args.dataset, args.outputs)
    rows = [row for row in rows if row['subset'] in ('train', 'validation')]
    phase = np.asarray([float(row['pdoa_deg']) for row in rows])
    expected = np.asarray([float(row['eta_expected_deg']) for row in rows])
    heights = np.asarray([float(row['height_difference_m']) for row in rows])
    tags = np.asarray([row['tag'] for row in rows])
    if not np.isfinite(phase).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite phase input or target')
    predictions = {'raw': wrap(phase), 'tag_harmonic': np.empty(len(rows)),
                   'tag_constant': np.empty(len(rows))}
    folds = []
    for held in sorted(set(heights)):
        models = []
        for tag in sorted(set(tags)):
            train = (heights != held) & (tags == tag)
            evaluate = (heights == held) & (tags == tag)
            if not train.any() or not evaluate.any():
                raise ValueError(f'Missing tag/height samples: {tag}, {held}')
            subsets = np.full(train.sum(), 'train', dtype=object)
            model = fit_trigonometric(phase[train], expected[train], subsets)
            correction, _, _ = predict_correction(phase[evaluate], model)
            constant = circular_summary(expected[train])['circular_bias_deg']
            predictions['tag_harmonic'][evaluate] = wrap(phase[evaluate] - correction)
            predictions['tag_constant'][evaluate] = constant
            models.append(dict(tag=tag, training_samples=int(train.sum()),
                               evaluation_samples=int(evaluate.sum()),
                               harmonic=model, constant_phase_deg=constant))
        folds.append(dict(held_height_m=float(held), models=models))
    metrics = []
    for name, prediction in predictions.items():
        for height in ('all', *sorted(set(heights))):
            select = np.ones(len(rows), dtype=bool) if height == 'all' else heights == height
            metrics.append(dict(method=name, held_height_m=height,
                                **circular_summary(wrap(prediction[select] - expected[select]))))
    destination = args.outputs / 'pdoa_tag_diagnostic_loho'
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / 'metrics.csv', metrics)
    write_csv(destination / 'predictions.csv', [dict(csv_line=row['csv_line'], tag=tags[i],
              held_height_m=heights[i], expected_phase_deg=expected[i],
              **{f'{name}_phase_deg': values[i] for name, values in predictions.items()})
              for i, row in enumerate(rows)])
    manifest = dict(protocol='leave_one_height_out', source_sha256=digest, split=split,
                    pool='train + validation; original grouped test excluded',
                    quality_policy='All finite phase observations; no distance-quality filter',
                    diagnostic='Tag identity supplies the fixed bearing; constant ignores measured phase',
                    folds=folds)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for result in metrics:
        if result['held_height_m'] == 'all':
            print(f"{result['method']}: n={result['samples']}, MAE={result['mae_deg']:.4f} deg, "
                  f"RMSE={result['rmse_deg']:.4f} deg")
    print(f'Outputs: {destination}')


if __name__ == '__main__':
    main()
