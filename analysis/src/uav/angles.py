"""Circular-angle helpers shared by UAV bearing and PDoA diagnostics."""

from __future__ import annotations

import math

import numpy as np


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    """Wrap degrees to the half-open interval [-180, 180)."""
    return (values + 180.0) % 360.0 - 180.0


def circular_mean_degrees(values: np.ndarray) -> float:
    """Return the circular mean of degree-valued observations."""
    radians = np.radians(values)
    return float(math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())))
