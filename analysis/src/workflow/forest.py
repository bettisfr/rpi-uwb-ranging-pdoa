"""Fixed-parameter, single-feature random-forest distance baseline."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .offset import is_distance_enabled


PARAMETERS = dict(n_estimators=100, max_depth=12, min_samples_leaf=20,
                  random_state=42, n_jobs=2, criterion='squared_error')


def fit_forest(rows):
    """Fit measured horizontal distance -> true distance on training only."""
    pairs = [(float(r['estimated_ground_distance_m']), float(r['target_ground_distance_m']))
             for r in rows if r['subset'] == 'train' and is_distance_enabled(r)]
    if not pairs or not np.isfinite(pairs).all():
        raise ValueError('Need finite eligible training observations')
    data = np.asarray(pairs)
    model = RandomForestRegressor(**PARAMETERS)
    model.fit(data[:, :1], data[:, 1])
    return model
