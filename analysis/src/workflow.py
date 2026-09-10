#!/usr/bin/env python3
"""Prepare reproducible benchmark contracts for PDoA correction experiments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from workflow.contracts import MODEL_CATALOG, SPLIT_DESCRIPTIONS
from workflow.export import prepare_output_contracts
from workflow.features import validate_feature_set
from workflow.splits import validate_split_name


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
DEFAULT_CONFIG = ANALYSIS / "configs" / "static_benchmark.json"
OUTPUTS = ANALYSIS / "outputs"


def load_config(path: Path) -> dict[str, object]:
    """Load and validate the versioned static benchmark definition."""
    with path.open() as source:
        config = json.load(source)
    for name, features in config["feature_sets"].items():
        if not isinstance(features, list):
            raise ValueError(f"feature set {name!r} must be a list")
        validate_feature_set(features)
    for split in config["splits"]:
        validate_split_name(split)
    unknown_models = set(config["models"]) - set(MODEL_CATALOG)
    if unknown_models:
        raise ValueError(f"unknown model(s): {sorted(unknown_models)}")
    return config


def write_plan(config: dict[str, object], output_directory: Path) -> Path:
    """Export one planned benchmark combination per row for review and plotting."""
    plan_path = output_directory / "benchmark_plan.csv"
    with plan_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("campaign", "feature_set", "target_name", "model_name", "model_stage", "model_status", "split_name", "split_description"),
        )
        writer.writeheader()
        for feature_set in config["feature_sets"]:
            for target in config["targets"]:
                for model_name in config["models"]:
                    model = MODEL_CATALOG[model_name]
                    for split_name in config["splits"]:
                        writer.writerow(
                            {
                                "campaign": config["campaign"],
                                "feature_set": feature_set,
                                "target_name": target,
                                "model_name": model_name,
                                "model_stage": model.stage,
                                "model_status": model.status,
                                "split_name": split_name,
                                "split_description": SPLIT_DESCRIPTIONS[split_name],
                            }
                        )
    return plan_path


def write_manifest(config_path: Path, config: dict[str, object], output_directory: Path) -> Path:
    """Record immutable benchmark intent next to the generated CSV contracts."""
    manifest_path = output_directory / "benchmark_manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "dataset": config["dataset"],
        "status": "skeleton_only",
        "note": "No measurements were fitted and no model predictions were generated.",
    }
    with manifest_path.open("w") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare", action="store_true", help="create stable CSV output contracts")
    parser.add_argument("--plan", action="store_true", help="export the planned benchmark combinations")
    args = parser.parse_args()
    if not args.prepare and not args.plan:
        parser.error("choose --prepare, --plan, or both")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.prepare:
        for path in prepare_output_contracts(OUTPUTS):
            print(f"Prepared {path.relative_to(ROOT)}")
    if args.plan:
        print(f"Wrote {write_plan(config, OUTPUTS).relative_to(ROOT)}")
        print(f"Wrote {write_manifest(config_path, config, OUTPUTS).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
