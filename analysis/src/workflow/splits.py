"""Split registry for honest static generalization and UAV transfer tests."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .contracts import SPLIT_DESCRIPTIONS


def validate_split_name(name: str) -> None:
    """Validate requested benchmark split names."""
    if name not in SPLIT_DESCRIPTIONS:
        raise ValueError(f"unknown split {name!r}; choose from {sorted(SPLIT_DESCRIPTIONS)}")


def build_grouped_split(rows: list[dict[str, str]], *, seed: int = 42,
                        validation_fraction: float = 0.2,
                        test_fraction: float = 0.2) -> list[dict]:
    """Assign whole physical conditions, stratified by height, to three subsets.

    Input rows are run/tag counts from a successful static audit. All tags and
    repeated acquisitions at the same (height, distance, rotation) stay together.
    Fractions refer to conditions, not individual measurements. Sorting before
    seeded shuffling makes assignments independent of input-file row order.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    factors = ("height_difference_m", "target_ground_distance_m", "tag_rotation_deg")
    groups = defaultdict(set)
    run_keys = {}
    for row in rows:
        key = tuple(float(row[field]) for field in factors)
        if not all(math.isfinite(value) for value in key):
            raise ValueError("Nonfinite grouping factor")
        run = (row["experiment_id"], row["run_file"])
        if run_keys.setdefault(run, key) != key:
            raise ValueError(f"Run has inconsistent physical conditions: {run}")
        groups[key[0]].add(key)
    if not groups:
        raise ValueError("No groups to split")
    assignments = {}
    rng = random.Random(seed)
    for height, keys in sorted(groups.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_validation = max(1, round(len(keys) * validation_fraction))
        n_test = max(1, round(len(keys) * test_fraction))
        if n_validation + n_test >= len(keys):
            raise ValueError(f"Too few groups at height {height} for three nonempty subsets")
        for index, key in enumerate(keys):
            assignments[key] = ("test" if index < n_test else "validation"
                                if index < n_test + n_validation else "train")
    output = []
    for row in sorted(rows, key=lambda r: (r["experiment_id"], r["run_file"], r["tag"])):
        key = tuple(float(row[field]) for field in factors)
        output.append({
            "split_name": "grouped_run", "fold_id": "holdout_0",
            "subset": assignments[key],
            "group_id": "h{:g}_d{:g}_rotation{:g}".format(*key),
            **{field: row[field] for field in ("experiment_id", "run_file", "tag", *factors)},
            "samples": int(row["samples"]),
        })
    return output


def export_grouped_split(output: Path, audit: dict, settings: dict) -> list[dict]:
    """Export run/tag assignments, actual counts, and reproducibility metadata."""
    if audit["status"] != "passed":
        raise ValueError("A successful current dataset audit is required")
    with (output / "static_run_inventory.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assigned = build_grouped_split(rows, **settings)
    with (output / "static_grouped_assignments.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(assigned[0]))
        writer.writeheader()
        writer.writerows(assigned)
    summary = []
    heights = sorted({float(row["height_difference_m"]) for row in assigned})
    for height in [None, *heights]:
        for subset in ("train", "validation", "test"):
            selected = [row for row in assigned if row["subset"] == subset and
                        (height is None or float(row["height_difference_m"]) == height)]
            summary.append({
                "height_difference_m": "all" if height is None else height,
                "subset": subset,
                "groups": len({row["group_id"] for row in selected}),
                "runs": len({(row["experiment_id"], row["run_file"]) for row in selected}),
                "tag_run_pairs": len(selected),
                "samples": sum(row["samples"] for row in selected),
            })
    with (output / "static_grouped_summary.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "split_name": "grouped_run", "fold_id": "holdout_0", **settings,
        "source": audit["source"], "source_sha256": audit["source_sha256"],
        "group_fields": ["height_difference_m", "target_ground_distance_m", "tag_rotation_deg"],
        "stratify_by": "height_difference_m",
        "fraction_unit": "physical conditions; sample fractions can differ",
        "algorithm": "v1: sorted conditions per height, Python Random(seed), test then validation then train",
        "assignment_join_keys": ["experiment_id", "run_file", "tag"],
        "note": "Fit preprocessing and models on train only; tune on validation; reserve test for final evaluation. Coplanar studies use the h=0 subset of these same assignments.",
    }
    (output / "static_grouped_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary
