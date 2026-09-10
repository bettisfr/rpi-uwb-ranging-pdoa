"""Periodic regression of a circular residual from the measured phase only."""
import numpy as np


def design(phase, harmonics=2):
    phase = np.deg2rad(np.asarray(phase, dtype=float))
    if phase.ndim != 1 or not np.isfinite(phase).all():
        raise ValueError('Need a finite phase vector')
    return np.column_stack([np.ones_like(phase)] +
                           [f(k*phase) for k in range(1, harmonics+1) for f in (np.sin, np.cos)])


def fit_trigonometric(phase, expected, subsets):
    from .phase import fit_phase_offset, wrap
    phase, expected = np.asarray(phase), np.asarray(expected)
    selected = np.asarray(subsets) == 'train'
    X = design(phase[selected])
    residual = np.deg2rad(wrap(phase[selected]-expected[selected]))
    Y = np.column_stack((np.cos(residual), np.sin(residual)))
    if not len(Y) or not np.isfinite(Y).all():
        raise ValueError('Need finite training targets')
    coefficients, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if rank != X.shape[1]:
        raise ValueError('Insufficient training phase diversity')
    return dict(harmonics=2, coefficients=coefficients.tolist(),
                basis=['1', 'sin(eta)', 'cos(eta)', 'sin(2*eta)', 'cos(2*eta)'],
                target_components=['cos(residual)', 'sin(residual)'],
                fallback_offset_deg=fit_phase_offset(phase, expected, subsets), minimum_vector_norm=1e-8)


def predict_correction(phase, model):
    vectors = design(phase, model['harmonics']) @ np.asarray(model['coefficients'])
    magnitude = np.linalg.norm(vectors, axis=1)
    correction = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    fallback = magnitude < model['minimum_vector_norm']
    correction[fallback] = model['fallback_offset_deg']
    return correction, magnitude, fallback
