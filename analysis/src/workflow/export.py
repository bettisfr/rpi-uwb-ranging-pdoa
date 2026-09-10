"""Create stable, header-only CSV interfaces for later benchmark outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from .contracts import METRIC_COLUMNS, POSITION_COLUMNS, PREDICTION_COLUMNS


def ensure_csv(path: Path, columns: tuple[str, ...]) -> Path:
    """Create a CSV with its stable header, without replacing existing output."""
    if path.exists():
        return path
    with path.open("w", newline="") as destination:
        csv.writer(destination).writerow(columns)
    return path


def prepare_output_contracts(output_directory: Path) -> list[Path]:
    """Prepare CSV artifacts consumed later by plots and manuscript tables."""
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        ensure_csv(output_directory / "predictions.csv", PREDICTION_COLUMNS),
        ensure_csv(output_directory / "metrics.csv", METRIC_COLUMNS),
        ensure_csv(output_directory / "position_metrics.csv", POSITION_COLUMNS),
    ]
