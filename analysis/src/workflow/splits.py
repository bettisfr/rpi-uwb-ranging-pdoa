"""Split registry for honest static generalization and UAV transfer tests."""

from __future__ import annotations

from .contracts import SPLIT_DESCRIPTIONS


def validate_split_name(name: str) -> None:
    """Validate requested benchmark split names."""
    if name not in SPLIT_DESCRIPTIONS:
        raise ValueError(f"unknown split {name!r}; choose from {sorted(SPLIT_DESCRIPTIONS)}")


def build_splits(*_: object) -> None:
    """Placeholder for deterministic fold construction.

    Fold IDs, held-out groups, and random seeds will be exported in the run
    manifest before models are trained.
    """
    raise NotImplementedError("Split construction has not been implemented yet.")
