#!/usr/bin/env python3
"""Does the static PDoA-stability window gate transfer to the flying node?

Tags are fixed and the node moves, so a window estimate is the mean of the
per-sample tag estimates in the local frame.  Gate thresholds are frozen from
the static study.  Pre-registered in analysis/README.md ("UAV window gate").
"""

from __future__ import annotations

import csv
import json

import numpy as np

from evaluate_uav_bridge import blocks
from evaluate_uav_range import timestamp
from uav.config import ANALYSIS
from workflow.bootstrap import mean_interval
from workflow.phase import circular_summary
from workflow.position_benchmark import write_csv


SOURCE = ANALYSIS / "outputs" / "uav_pdoa_position" / "predictions.csv"
OUTPUT = ANALYSIS / "outputs" / "uav_window_gate"
WINDOW_SIZES = (10, 20, 30)
PRIMARY_K = 20
MAX_SAMPLE_GAP_S = .5
RANGE_GATE_M, ELEVATION_GATE_DEG = 25., 45.


def windows(rows: list[dict], size: int) -> list[dict]:
    """Non-overlapping, gap-free windows per training direction, flight and tag."""
    series: dict[tuple, list[dict]] = {}
    for row in rows:
        series.setdefault((row["train_flight"], row["flight"], row["tag"]), []).append(row)
    records = []
    for (train_flight, flight, tag), samples in series.items():
        samples.sort(key=lambda row: row["seconds"])
        for start in range(0, len(samples) - size + 1, size):
            window = samples[start:start + size]
            seconds = np.asarray([row["seconds"] for row in window])
            if np.diff(seconds).max() > MAX_SAMPLE_GAP_S:
                continue
            x_hat = float(np.mean([row["tag_x_hat_m"] for row in window]))
            y_hat = float(np.mean([row["tag_y_hat_m"] for row in window]))
            records.append(dict(
                train_flight=train_flight, flight=flight, tag=tag, start_s=float(seconds[0]),
                span_s=float(seconds[-1] - seconds[0]),
                pdoa_std_deg=circular_summary(np.asarray([row["pdoa_deg"] for row in window]))["circular_std_deg"],
                mean_range_m=float(np.mean([row["raw_range_m"] for row in window])),
                mean_elevation_deg=float(np.mean([row["elevation_hat_deg"] for row in window])),
                position_error_m=float(np.hypot(x_hat - window[0]["tag_x_m"], y_hat - window[0]["tag_y_m"])),
                single_sample_mae_m=float(np.mean([row["position_error_m"] for row in window]))))
    return records


def main() -> None:
    with SOURCE.open(newline="") as source:
        raw = list(csv.DictReader(source))
    if "tag_x_hat_m" not in raw[0]:
        raise SystemExit("Rerun evaluate_uav_pdoa_position.py: per-sample tag estimates are missing")
    origin = min(timestamp(row["uwb_time"]) for row in raw)
    rows = [dict(row, seconds=(timestamp(row["uwb_time"]) - origin).total_seconds(),
                 **{k: float(row[k]) for k in ("pdoa_deg", "raw_range_m", "elevation_hat_deg", "tag_x_hat_m",
                                               "tag_y_hat_m", "tag_x_m", "tag_y_m", "position_error_m")})
            for row in raw]
    summary = []
    for size in WINDOW_SIZES:
        records = windows(rows, size)
        for train_flight in sorted({r["train_flight"] for r in records}):
            part = [r for r in records if r["train_flight"] == train_flight]
            error = np.asarray([r["position_error_m"] for r in part])
            single = np.asarray([r["single_sample_mae_m"] for r in part])
            std = np.asarray([r["pdoa_std_deg"] for r in part])
            near = (np.asarray([r["mean_range_m"] for r in part]) <= RANGE_GATE_M) & \
                   (np.asarray([r["mean_elevation_deg"] for r in part]) <= ELEVATION_GATE_DEG)
            groups = blocks([r["flight"] for r in part], [r["tag"] for r in part], [r["start_s"] for r in part])
            for gate, keep in (("all", np.ones(len(part), dtype=bool)), ("std<=10", std <= 10.),
                               ("std<=20", std <= 20.), ("range", near)):
                if keep.sum() < 10:
                    continue
                mae, (low, high), _ = mean_interval(error[keep], groups[keep])
                summary.append(dict(window_size=size, train_flight=train_flight, gate=gate, windows=int(keep.sum()),
                                    coverage=float(keep.mean()), position_mae_m=mae, ci_low_m=low, ci_high_m=high,
                                    p90_m=float(np.quantile(error[keep], .9, method="inverted_cdf")),
                                    same_windows_single_sample_mae_m=float(single[keep].mean()),
                                    median_pdoa_std_deg=float(np.median(std[keep])),
                                    median_span_s=float(np.median([r["span_s"] for r, k in zip(part, keep) if k]))))
    primary = [r for r in summary if r["window_size"] == PRIMARY_K and r["gate"] == "std<=20"]
    source_manifest = json.loads(SOURCE.with_name("manifest.json").read_text())
    verdict = dict(kind="descriptive window assessment; no success threshold",
                   uav_mae_m={r["train_flight"]: r["position_mae_m"] for r in primary})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "metrics.csv", summary)
    (OUTPUT / "manifest.json").write_text(json.dumps(dict(
        status="complete", source=str(SOURCE.relative_to(ANALYSIS)), window_sizes=list(WINDOW_SIZES),
        primary_chain=source_manifest["primary_chain"],
        static_source_sha256=source_manifest["static_references"]["static_source_sha256"],
        max_sample_gap_s=MAX_SAMPLE_GAP_S, estimate="mean of per-sample tag estimates in the local frame",
        verdict=verdict), indent=2) + "\n")
    for row in summary:
        print(f"K={row['window_size']} train={row['train_flight'][-6:]} {row['gate']}: n={row['windows']} "
              f"cov={row['coverage']:.2f} MAE={row['position_mae_m']:.2f} m [{row['ci_low_m']:.2f}, {row['ci_high_m']:.2f}] "
              f"P90={row['p90_m']:.2f} single={row['same_windows_single_sample_mae_m']:.2f} "
              f"std_med={row['median_pdoa_std_deg']:.1f} span={row['median_span_s']:.1f}s")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
