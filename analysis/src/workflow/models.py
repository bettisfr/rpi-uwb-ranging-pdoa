"""Model registry; fitting stays intentionally unimplemented for now."""

from __future__ import annotations

from .contracts import MODEL_CATALOG


def get_model_spec(name: str):
    """Return a registered model family or fail with the available choices."""
    try:
        return MODEL_CATALOG[name]
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_CATALOG)}") from error


def fit_model(*_: object) -> None:
    """Placeholder for fold-local preprocessing and model fitting."""
    raise NotImplementedError("Model fitting has not been implemented yet.")
