"""Feature-set definitions and leakage guards for correction benchmarks."""

from __future__ import annotations


GROUND_TRUTH_COLUMNS = frozenset(
    {
        "target_ground_distance_m",
        "gt_bearing_deg",
        "gt_x_m",
        "gt_y_m",
        "gt_range_3d_m",
    }
)


def validate_feature_set(features: list[str]) -> None:
    """Reject target-side geometry before a model can be fitted."""
    leaked = GROUND_TRUTH_COLUMNS.intersection(features)
    if leaked:
        raise ValueError(f"ground-truth columns cannot be model features: {sorted(leaked)}")


def build_features(*_: object) -> None:
    """Placeholder for numeric/cyclic feature construction.

    The implementation will encode angular observables cyclically and fit every
    preprocessing transform within each training fold only.
    """
    raise NotImplementedError("Feature construction has not been implemented yet.")
