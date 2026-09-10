"""Common estimators for single-feature horizontal-distance comparisons."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


class OffsetRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.offset_ = float(np.mean(np.asarray(y) - np.asarray(X)[:, 0]))
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] + self.offset_


class WeightedDuplicateSVR(RegressorMixin, BaseEstimator):
    """Compress exact repeated (x,y) pairs; weights preserve the SVR objective.

    All observations, including extreme values, retain their full multiplicity.
    Feature scaling is fitted on the original training rows before compression.
    No rounding, subsampling, or target-based rejection is performed.
    """
    def __init__(self, parameters=None, allow_partial=False):
        self.parameters = parameters
        self.allow_partial = allow_partial

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        transformed = self.scaler_.transform(X)
        unique, counts = np.unique(np.column_stack((transformed, y)), axis=0, return_counts=True)
        self.training_samples_ = len(y)
        self.unique_pairs_ = len(unique)
        self.regressor_ = SVR(**(self.parameters or {}))
        self.regressor_.fit(unique[:, :-1], unique[:, -1], sample_weight=counts.astype(float))
        if self.regressor_.fit_status_ != 0 and not self.allow_partial:
            raise RuntimeError('SVR did not converge')
        return self

    def predict(self, X):
        return self.regressor_.predict(self.scaler_.transform(X))


METHODS = ('constant_offset', 'linear_regression', 'quadratic_regression',
           'random_forest', 'svr', 'mlp')


def make_model(name, config):
    if name == 'constant_offset':
        return OffsetRegressor()
    if name == 'linear_regression':
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == 'quadratic_regression':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression())
    if name == 'random_forest':
        return RandomForestRegressor(random_state=config['seed'], **config[name])
    if name == 'svr':
        return WeightedDuplicateSVR(parameters=config[name])
    if name == 'mlp':
        parameters = {**config[name], 'hidden_layer_sizes': tuple(config[name]['hidden_layer_sizes'])}
        # Both input and target transformations are learned within fit(train).
        # No internal random validation split: stopping monitors training loss.
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(random_state=config['seed'], **parameters)),
            transformer=StandardScaler())
    raise ValueError(f'Unknown distance model: {name}')
