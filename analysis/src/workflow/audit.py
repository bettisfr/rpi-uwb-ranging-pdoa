"""Inspect normalized static data before assigning any training/test folds."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from .data import STATIC_REQUIRED_COLUMNS, validate_normalized_static_table


BEARINGS = (15, 35, 55, 75, 90, 105, 125, 145, 165)
HEIGHTS = (0, 2, 4)
DISTANCES = tuple(range(2, 31, 2))
ROTATIONS = (0, 90, 180, 270)
FACTORS = ("height_difference_m", "target_ground_distance_m", "tag_rotation_deg")
IDENTIFIERS = {"experiment_id", "run_file", "tag"}
NUMERIC = STATIC_REQUIRED_COLUMNS - IDENTIFIERS


def audit_static(path: Path, output: Path) -> dict:
    """Export observed runs, full-design coverage, and data-quality findings.

    Counts include every input row, including flagged rows. Errors are reported
    without silently cleaning data. Empty horizontal errors are allowed only
    for geometrically invalid projections (range < height).
    """
    validate_normalized_static_table(path)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    run_counts = Counter()
    run_factors = {}
    config_runs = {}
    error_counts = Counter()
    examples = []
    total = invalid_projection = 0

    def flag(line: int, code: str, detail: str) -> None:
        error_counts[code] += 1
        if len(examples) < 30:
            examples.append({"csv_line": line, "code": code, "detail": detail})

    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for line, row in enumerate(reader, start=2):
            total += 1
            if None in row or any(value is None for value in row.values()):
                flag(line, "malformed_row", "Column count differs from header")
                continue
            if any(not row[name].strip() for name in IDENTIFIERS):
                flag(line, "missing_identifier", "Empty experiment, run, or tag")
                continue
            values = {}
            for name in sorted(NUMERIC):
                if name == "ground_distance_error_m" and row[name] == "":
                    continue
                try:
                    value = float(row[name])
                    if not math.isfinite(value):
                        raise ValueError
                    values[name] = value
                except ValueError:
                    flag(line, "invalid_numeric", name)
            if not all(name in values for name in FACTORS):
                continue
            factors = tuple(values[name] for name in FACTORS)
            h, distance, rotation = factors
            tag = row["tag"]
            if h not in HEIGHTS or distance not in DISTANCES or rotation not in ROTATIONS:
                flag(line, "unexpected_configuration", str(factors))
            expected_tags = {f"dw{i:02}": bearing for i, bearing in enumerate(BEARINGS)}
            if tag not in expected_tags or values.get("gt_bearing_deg") != expected_tags.get(tag):
                flag(line, "tag_bearing_mismatch", tag)
            run = (row["experiment_id"], row["run_file"])
            previous = run_factors.setdefault(run, factors)
            if previous != factors:
                flag(line, "mixed_run_factors", str(run))
            key = (*factors, tag)
            counts[key] += 1
            run_counts[(*run, tag)] += 1
            config_runs.setdefault(key, set()).add(run)
            if "range_m" in values:
                if values["range_m"] < 0:
                    flag(line, "negative_range", row["range_m"])
                if values["range_m"] < h:
                    invalid_projection += 1
                    if row["ground_distance_error_m"] != "":
                        flag(line, "invalid_projection_has_target", "Expected empty horizontal error")
                elif "ground_distance_error_m" not in values:
                    flag(line, "missing_ground_target", "Valid projection has no horizontal error")
                else:
                    expected = math.sqrt(values["range_m"] ** 2 - h ** 2) - distance
                    if not math.isclose(values["ground_distance_error_m"], expected, abs_tol=1e-8):
                        flag(line, "inconsistent_ground_target", "Horizontal error differs from geometry")

    fields = (*FACTORS, "tag", "expected_bearing_deg", "samples", "source_runs", "coverage")
    with (output / "static_configuration_inventory.csv").open("w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(fields)
        expected_keys = {
            (h, d, rotation, f"dw{i:02}")
            for h, d, rotation, i in itertools.product(HEIGHTS, DISTANCES, ROTATIONS, range(9))
        }
        for key in sorted(expected_keys | set(counts)):
            tag = key[-1]
            n = counts[key]
            coverage = "missing" if n == 0 else "partial" if n < 200 else "complete" if n == 200 else "above_target"
            bearing = dict(zip((f"dw{i:02}" for i in range(9)), BEARINGS)).get(tag, "")
            writer.writerow((*key, bearing, n, len(config_runs.get(key, ())), coverage))

    with (output / "static_run_inventory.csv").open("w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(("experiment_id", "run_file", *FACTORS, "tag", "samples"))
        for (experiment, run, tag), n in sorted(run_counts.items()):
            writer.writerow((experiment, run, *run_factors[(experiment, run)], tag, n))

    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "source": str(path.resolve()),
        "source_sha256": digest,
        "rows": total,
        "runs": len(run_factors),
        "expected_configurations": len(expected_keys),
        "observed_configurations": len(counts),
        "missing_configurations": len(expected_keys - set(counts)),
        "invalid_horizontal_projections": invalid_projection,
        "rows_by_height": {str(h): sum(n for key, n in counts.items() if key[0] == h) for h in HEIGHTS},
        "error_counts": dict(error_counts),
        "error_examples": examples,
        "status": "failed" if error_counts or not total else "passed",
        "scope": "Normalized static CSV only; source-file completeness and duplicate acquisition events are not verified.",
        "grouping": "Keep experiment_id + run_file together; repeated physical configurations must share the same split.",
    }
    (output / "static_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
