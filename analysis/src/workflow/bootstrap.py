"""Grouped bootstrap: resample whole physical configurations, not samples.

The ~200 samples of one tag in one run are strongly correlated, so the unit of
replication is the configuration (height, distance, rotation, tag).  Group sums
are precomputed once; each replicate is a multinomial reweighting of groups.
"""

from __future__ import annotations

import numpy as np


N_BOOT = 1000
SEED = 42


def configuration_keys(tag, height, distance, rotation):
    """One label per physical configuration; repeated runs fall in the same group."""
    return np.asarray([f"{t}|{h:g}|{d:g}|{r:g}" for t, h, d, r in zip(tag, height, distance, rotation)])


def bootstrap_weights(n_groups, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_groups) multiplicities of each group in each replicate."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1. / n_groups), size=n_boot).astype(float)


def group_sums(values, groups):
    """Per-group sums of `values` (n, k) and per-group counts."""
    _, index = np.unique(groups, return_inverse=True)
    sums = np.zeros((index.max() + 1, values.shape[1]))
    np.add.at(sums, index, values)
    return sums, np.bincount(index).astype(float)


def bootstrap_means(values, groups, n_boot=N_BOOT, seed=SEED):
    """Bootstrap distribution of the sample mean of `values` under group resampling."""
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1
    sums, counts = group_sums(values.reshape(len(values), -1), groups)
    weights = bootstrap_weights(len(counts), n_boot, seed)
    means = (weights @ sums) / (weights @ counts)[:, None]
    return means[:, 0] if one_dimensional else means


def percentile_interval(samples, level=.95):
    low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return float(low), float(high)


def mean_interval(values, groups, n_boot=N_BOOT, seed=SEED):
    """Point estimate, 95% interval and bootstrap samples of a mean."""
    samples = bootstrap_means(values, groups, n_boot, seed)
    return float(np.mean(values)), percentile_interval(samples), samples


def circular_mean_interval(angles_deg, groups, n_boot=N_BOOT, seed=SEED):
    """Circular mean in degrees with an interval centred on the point estimate."""
    radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
    vectors = np.column_stack((np.cos(radians), np.sin(radians)))
    estimate = float(np.rad2deg(np.arctan2(vectors[:, 1].mean(), vectors[:, 0].mean())))
    boot = bootstrap_means(vectors, groups, n_boot, seed)
    samples = np.rad2deg(np.arctan2(boot[:, 1], boot[:, 0]))
    return estimate, wrapped_interval(estimate, samples), samples


def wrapped_interval(estimate, samples):
    """Percentile interval of angular samples, unwrapped around the estimate."""
    deviation = (np.asarray(samples) - estimate + 180.) % 360. - 180.
    low, high = percentile_interval(deviation)
    return estimate + low, estimate + high
