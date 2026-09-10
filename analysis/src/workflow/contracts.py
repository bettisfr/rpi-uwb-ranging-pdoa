"""Stable contracts shared by dataset adapters, models, metrics, and plots."""

from __future__ import annotations

from dataclasses import dataclass


PREDICTION_COLUMNS = (
    "run_id",
    "campaign",
    "split_name",
    "fold_id",
    "model_name",
    "feature_set",
    "target_name",
    "sample_id",
    "group_id",
    "y_true",
    "y_pred",
    "residual",
)

METRIC_COLUMNS = (
    "run_id",
    "campaign",
    "split_name",
    "fold_id",
    "model_name",
    "feature_set",
    "target_name",
    "samples",
    "bias",
    "mae",
    "rmse",
    "std",
    "r2",
    "circular_mae_deg",
    "circular_std_deg",
)

POSITION_COLUMNS = (
    "run_id",
    "campaign",
    "split_name",
    "fold_id",
    "model_name",
    "feature_set",
    "sample_id",
    "position_error_raw_m",
    "position_error_corrected_m",
)


@dataclass(frozen=True)
class ModelSpec:
    """One registered model family and its intended role."""

    name: str
    stage: str
    status: str
    note: str


MODEL_CATALOG = {
    "constant_offset": ModelSpec("constant_offset", "baseline", "planned", "Per-training-fold signed offset."),
    "linear_regression": ModelSpec("linear_regression", "parametric", "planned", "Linear correction of a signed target."),
    "quadratic_regression": ModelSpec("quadratic_regression", "parametric", "planned", "Polynomial correction with controlled degree two."),
    "trigonometric_regression": ModelSpec("trigonometric_regression", "parametric", "planned", "Periodic PDoA-residual correction."),
    "random_forest": ModelSpec("random_forest", "machine_learning", "planned", "Nonlinear tabular baseline."),
    "support_vector_regression": ModelSpec("support_vector_regression", "machine_learning", "planned", "Kernel regression baseline."),
    "mlp_regression": ModelSpec("mlp_regression", "machine_learning", "planned", "Supervised neural baseline."),
    "autoencoder_representation": ModelSpec("autoencoder_representation", "representation", "experimental", "Optional denoising/representation stage, not a direct regressor."),
}


SPLIT_DESCRIPTIONS = {
    "random_diagnostic": "Optimistic diagnostic split; never the sole reported result.",
    "grouped_run": "Hold out complete acquisition runs/configurations.",
    "leave_one_distance_out": "Hold out one complete controlled radial distance.",
    "leave_one_height_out": "Hold out one complete node--tag height difference.",
    "uav_zero_shot": "Fit static data only and evaluate the untouched UAV campaign.",
    "uav_finetune_ablation": "Optional adaptation study with disjoint flight/time folds.",
}
