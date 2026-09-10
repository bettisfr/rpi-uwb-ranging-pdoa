"""Quadratic least squares using training-only centering and scaling."""

import numpy as np

from .offset import is_distance_enabled


def fit_quadratic(rows):
    pairs = [(float(r['estimated_ground_distance_m']), float(r['target_ground_distance_m']))
             for r in rows if r['subset'] == 'train' and is_distance_enabled(r)]
    if len(pairs) < 3:
        raise ValueError('Need at least three training observations')
    data = np.asarray(pairs)
    if not np.isfinite(data).all():
        raise ValueError('Training contains nonfinite values')
    x, y = data.T
    center, scale = float(x.mean()), float(x.std())
    if scale == 0:
        raise ValueError('Training input distance has zero variance')
    z = (x - center) / scale
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.column_stack((np.ones_like(z), z, z*z)), y, rcond=None)
    if rank != 3:
        raise ValueError('Need three distinct input distances for a quadratic fit')
    c0, c1, c2 = map(float, coefficients)
    return dict(center_m=center, scale_m=scale, coefficients=[c0, c1, c2],
                quadratic_per_m=c2 / scale**2,
                slope=c1 / scale - 2*c2*center / scale**2,
                intercept_m=c0 - c1*center/scale + c2*center**2/scale**2,
                numpy_version=np.__version__)


def predict_quadratic(x, model):
    z = (x - model['center_m']) / model['scale_m']
    c0, c1, c2 = model['coefficients']
    return c0 + z * (c1 + z*c2)
