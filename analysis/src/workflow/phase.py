"""Raw PDoA and a single training-fitted circular phase offset."""

import json
import numpy as np

from .linear import write_csv
from .offset import load_assigned_rows


def wrap(values):
    return (np.asarray(values, dtype=float) + 180.) % 360. - 180.


def circular_summary(errors):
    errors = wrap(errors)
    if not errors.size or not np.isfinite(errors).all():
        raise ValueError('Need nonempty finite phase residuals')
    vector = np.mean(np.exp(1j * np.deg2rad(errors)))
    length = min(1., float(abs(vector)))
    defined = length > 1e-12
    return dict(samples=len(errors),
                circular_bias_deg=float(wrap(np.rad2deg(np.angle(vector)))) if defined else None,
                circular_std_deg=float(np.rad2deg(np.sqrt(-2*np.log(length)))) if defined else None,
                resultant_length=length,
                mae_deg=float(np.abs(errors).mean()),
                rmse_deg=float(np.sqrt(np.mean(errors**2))),
                p90_absolute_error_deg=float(np.quantile(np.abs(errors), .9, method='inverted_cdf')))


def fit_phase_offset(phase, expected, subsets):
    selected = np.asarray(subsets) == 'train'
    result = circular_summary(wrap(np.asarray(phase)[selected] - np.asarray(expected)[selected]))
    if result['circular_bias_deg'] is None:
        raise ValueError('Training residuals have no defined circular mean')
    return result['circular_bias_deg']


def run_phase(dataset, output, *, trigonometric=False):
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [r for r in rows if r['subset'] in ('train', 'validation')]
    phase = np.array([float(r['pdoa_deg']) for r in rows])
    expected = np.array([float(r['eta_expected_deg']) for r in rows])
    # Validate target against the already adopted height-aware far-field model.
    reference = np.array([-180 * float(r['gt_x_m']) / float(r['gt_range_3d_m']) for r in rows])
    if not np.isfinite(phase).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite phase data')
    if not np.allclose(reference, expected, atol=1e-9, rtol=0):
        raise ValueError('Expected phase does not match the height-aware reference')
    subset = np.array([r['subset'] for r in rows])
    height = np.array([float(r['height_difference_m']) for r in rows])
    tag = np.array([r['tag'] for r in rows])
    offset = fit_phase_offset(phase, expected, subset)
    corrected = wrap(phase-offset)
    residuals = dict(raw=wrap(phase-expected), circular_offset=wrap(corrected-expected))
    if trigonometric:
        from .phase_trigonometric import fit_trigonometric, predict_correction
        trig_model = fit_trigonometric(phase, expected, subset)
        correction, vector_norm, fallback = predict_correction(phase, trig_model)
        trig_corrected = wrap(phase-correction)
        residuals['trigonometric'] = wrap(trig_corrected-expected)
    summary, details = [], []
    for name, errors in residuals.items():
        for part in ('train', 'validation'):
            for h in [None, *sorted(set(height))]:
                select = (subset == part) & (True if h is None else height == h)
                if select.any():
                    summary.append(dict(subset=part, height_difference_m='all' if h is None else h,
                                        method=name, **circular_summary(errors[select])))
                for t in sorted(set(tag)):
                    selected = select & (tag == t)
                    if selected.any():
                        details.append(dict(subset=part, height_difference_m='all' if h is None else h,
                                            tag=t, method=name, **circular_summary(errors[selected])))
    destination = output / ('pdoa_trigonometric' if trigonometric else 'pdoa_offset')
    destination.mkdir(exist_ok=True)
    predictions = []
    for i, row in enumerate(rows):
        predictions.append(dict(csv_line=row['csv_line'], group_id=row['group_id'], subset=row['subset'],
                                tag=row['tag'], height_difference_m=height[i],
                                expected_phase_deg=expected[i], raw_phase_deg=phase[i],
                                corrected_phase_deg=corrected[i], raw_residual_deg=residuals['raw'][i],
                                corrected_residual_deg=residuals['circular_offset'][i]))
        if trigonometric:
            predictions[-1].update(trigonometric_phase_deg=trig_corrected[i],
                                   trigonometric_residual_deg=residuals['trigonometric'][i],
                                   predicted_correction_deg=correction[i],
                                   predicted_vector_norm=vector_norm[i], fallback_used=bool(fallback[i]))
    write_csv(destination / 'predictions.csv', predictions)
    write_csv(destination / 'metrics.csv', summary)
    write_csv(destination / 'metrics_by_tag_height.csv', details)
    model = dict(model='circular_phase_offset', offset_deg=offset,
                 correction='wrap(pdoa_deg-offset_deg)', fit_subset='train',
                 training_samples=int(np.sum(subset == 'train')), source_sha256=digest,
                 split=split, reference='eta_expected_deg = -180 * gt_x_m / gt_range_3d_m',
                 scope='Raw phase, not geometric bearing; nominal half-wavelength far-field reference',
                 fit_objective='Circular mean residual (minimizes chordal loss, not angular MAE)',
                 eligibility='All finite phase records in train/validation, including invalid horizontal projections',
                 metric_definition='Residual wrapped into [-180,180); MAE/RMSE of shortest angular differences; no linear R2',
                 outlier_policy='No rejection or clipping', evaluated_subsets=['train', 'validation'])
    if trigonometric:
        model.update(model='trigonometric_phase_correction', trigonometric=trig_model,
                     input='pdoa_deg only', correction='wrap(eta-atan2(predicted_sin_residual,predicted_cos_residual))',
                     fit_objective='Least squares on cosine and sine of the training residual',
                     model_selection='Two harmonics fixed before evaluation; no validation tuning',
                     fallback_counts={part: int(fallback[subset == part].sum()) for part in ('train', 'validation')},
                     vector_norm_note='Unconstrained regression magnitude, not a confidence probability')
    (destination / 'model.json').write_text(json.dumps(model, indent=2)+'\n')
    return model, summary
