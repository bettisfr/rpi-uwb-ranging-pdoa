"""Initial RF/MLP phase-feature comparison, fixed models and grouped validation."""
import json
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .linear import write_csv
from .offset import load_assigned_rows
from .phase import wrap, circular_summary, fit_phase_offset
from .phase_trigonometric import fit_trigonometric, predict_correction


def phase_features(rows, enriched=False):
    phase = np.deg2rad([float(r['pdoa_deg']) for r in rows])
    columns = [np.sin(phase), np.cos(phase)]
    names = ['sin_measured_phase', 'cos_measured_phase']
    if enriched:
        rotation = np.deg2rad([float(r['tag_rotation_deg']) for r in rows])
        columns += [[float(r['range_m']) for r in rows],
                    [float(r['height_difference_m']) for r in rows],
                    np.sin(rotation), np.cos(rotation)]
        names += ['range_m', 'height_difference_m', 'sin_tag_rotation', 'cos_tag_rotation']
    X = np.column_stack(columns)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite observed features')
    return X, names


def run_phase_learning(dataset, output, *, svr_only=False):
    rows, split, digest = load_assigned_rows(dataset, output)
    rows = [r for r in rows if r['subset'] in ('train', 'validation')]
    subset = np.array([r['subset'] for r in rows])
    train, valid = subset == 'train', subset == 'validation'
    phase = np.array([float(r['pdoa_deg']) for r in rows])
    expected = np.array([float(r['eta_expected_deg']) for r in rows])
    reference = [-180*float(r['gt_x_m'])/float(r['gt_range_3d_m']) for r in rows]
    if not np.isfinite(expected).all() or not np.allclose(expected, reference, atol=1e-9, rtol=0):
        raise ValueError('Invalid phase reference')
    target_angle = np.deg2rad(wrap(phase-expected))
    Y = np.column_stack((np.cos(target_angle), np.sin(target_angle)))
    offset = fit_phase_offset(phase, expected, subset)
    trig = fit_trigonometric(phase, expected, subset)
    correction, _, _ = predict_correction(phase[valid], trig)
    estimates = dict(raw=wrap(phase[valid]), circular_offset=wrap(phase[valid]-offset),
                     trigonometric=wrap(phase[valid]-correction))
    destination = output / ('pdoa_svr' if svr_only else 'pdoa_learning')
    destination.mkdir(exist_ok=True)
    manifest = dict(status='running', source_sha256=digest, split=split,
                    sklearn_version=sklearn.__version__, numpy_version=np.__version__,
                    target='cos/sin(wrap(measured_phase-expected_phase))',
                    training_samples=int(train.sum()), validation_samples=int(valid.sum()),
                    phase_offset_deg=offset, trigonometric=trig, models=[],
                    outlier_policy='All finite phase records retained; no range eligibility filter',
                    protocol='Fit on training only; evaluate validation; original test excluded',
                    feature_assumption='Tag orientation and node height are known at inference',
                    selection='Fixed parameters, no validation tuning')
    path = destination / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    for enriched in (False, True):
        X, names = phase_features(rows, enriched)
        for kind in (('svr',) if svr_only else ('random_forest', 'mlp')):
            name = kind + ('_enriched' if enriched else '_phase_only')
            print(f'Fitting {name}', flush=True)
            if kind == 'random_forest':
                params = dict(n_estimators=100, max_depth=12, min_samples_leaf=20, random_state=42, n_jobs=2)
                model = RandomForestRegressor(**params)
            elif kind == 'mlp':
                params = dict(hidden_layer_sizes=(32, 16), batch_size=2048, max_iter=200,
                              tol=1e-5, n_iter_no_change=20, early_stopping=False, random_state=42)
                model = make_pipeline(StandardScaler(), MLPRegressor(**params))
            else:
                from sklearn.multioutput import MultiOutputRegressor
                from .distance_models import WeightedDuplicateSVR
                params = dict(kernel='rbf', C=1., epsilon=.05, gamma='scale',
                              tol=.001, max_iter=100000, cache_size=512)
                model = MultiOutputRegressor(WeightedDuplicateSVR(params, allow_partial=True), n_jobs=1)
            started = time.monotonic()
            try:
                with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(X[train], Y[train])
                    vectors = model.predict(X[valid])
            except KeyboardInterrupt:
                manifest.update(status='interrupted', interrupted_model=name)
                path.write_text(json.dumps(manifest, indent=2)+'\n')
                raise
            correction = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
            fallback = np.linalg.norm(vectors, axis=1) < 1e-8
            correction[fallback] = offset
            estimates[name] = wrap(phase[valid]-correction)
            joblib.dump(model, destination / f'{name}.joblib')
            item = dict(name=name, features=names, parameters=params, fallback_count=int(fallback.sum()),
                        elapsed_s=time.monotonic()-started, warnings=[str(w.message) for w in caught])
            if kind == 'mlp':
                item['iterations'] = int(model.named_steps['mlpregressor'].n_iter_)
            if kind == 'svr':
                item['fit_status'] = [int(e.regressor_.fit_status_) for e in model.estimators_]
                item['iterations'] = [int(e.regressor_.n_iter_) for e in model.estimators_]
                item['unique_training_pairs'] = [e.unique_pairs_ for e in model.estimators_]
                item['converged'] = all(s == 0 for s in item['fit_status'])
                print(f"{name}: converged={item['converged']}, iterations={item['iterations']}", flush=True)
            manifest['models'].append(item)
            path.write_text(json.dumps(manifest, indent=2)+'\n')
    height = np.array([float(r['height_difference_m']) for r in rows])[valid]
    tags = np.array([r['tag'] for r in rows])[valid]
    summary, details = [], []
    for name, estimate in estimates.items():
        error = wrap(estimate-expected[valid])
        for h in [None, *sorted(set(height))]:
            selected = np.ones(len(height), dtype=bool) if h is None else height == h
            summary.append(dict(subset='validation', method=name, height_difference_m='all' if h is None else h,
                                **circular_summary(error[selected])))
            for tag in sorted(set(tags)):
                select = selected & (tags == tag)
                if select.any():
                    details.append(dict(subset='validation', method=name, height_difference_m='all' if h is None else h,
                                        tag=tag, **circular_summary(error[select])))
    prediction_rows = []
    for k, i in enumerate(np.flatnonzero(valid)):
        prediction_rows.append(dict(csv_line=rows[i]['csv_line'], group_id=rows[i]['group_id'],
                                    height_difference_m=height[k], tag=tags[k], expected_phase_deg=expected[i],
                                    **{name+'_deg': values[k] for name, values in estimates.items()}))
    write_csv(destination / 'predictions.csv', prediction_rows)
    write_csv(destination / 'metrics.csv', summary)
    write_csv(destination / 'metrics_by_tag_height.csv', details)
    manifest['status'] = ('complete_with_partial_fits' if any(not m.get('converged', True) for m in manifest['models']) else 'complete')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return summary
