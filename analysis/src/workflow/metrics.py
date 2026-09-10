"""Metric registry for model evaluation and paper-facing CSV exports."""

from __future__ import annotations


SCALAR_METRICS = ("bias", "mae", "rmse", "std", "r2")
CIRCULAR_METRICS = ("circular_mae_deg", "circular_std_deg")
POSITION_METRICS = ("position_error_raw_m", "position_error_corrected_m")


def evaluate_predictions(*_: object) -> None:
    """Placeholder for scalar, circular, and derived-position evaluation."""
    raise NotImplementedError("Metric evaluation has not been implemented yet.")
