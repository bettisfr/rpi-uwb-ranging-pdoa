#!/usr/bin/env python3
"""Recompute UAV local coordinates offline from RTK latitude/longitude.

The local origin is O and the positive y-axis is the RTK baseline O->Q4.
Coordinates use an independent WGS84 ECEF-to-ENU implementation, rather than
the values generated on-the-fly by the web logger.
"""

from __future__ import annotations

import csv
import json
import math

from uav.config import ANALYSIS, UAV_DATA
from uav.coordinates import enu, local_xy

UAV_DIRECTORY = UAV_DATA
OUTPUT = ANALYSIS / "csv" / "uav_local_frame_recalculation.csv"
SUMMARY = ANALYSIS / "csv" / "uav_local_frame_recalculation_summary.csv"


def main() -> None:
    reference = json.loads((UAV_DIRECTORY / "reference.json").read_text())
    rows, dx, dy = [], [], []
    for path in sorted(UAV_DIRECTORY.glob("drone_*.csv")):
        with path.open(newline="") as source:
            for line, row in enumerate(csv.DictReader(source), 2):
                if row["rtk_status"] != "50":
                    continue
                x_m, y_m = local_xy(reference, float(row["rtk_lat_deg"]), float(row["rtk_lon_deg"]))
                logged_x, logged_y = float(row["gt_x_m"]), float(row["gt_y_m"])
                dx.append(x_m - logged_x)
                dy.append(y_m - logged_y)
                rows.append(dict(
                    flight=path.stem, csv_line=line, rtk_lat_deg=row["rtk_lat_deg"], rtk_lon_deg=row["rtk_lon_deg"],
                    offline_x_m=x_m, offline_y_m=y_m, logged_x_m=logged_x, logged_y_m=logged_y,
                    delta_x_m=x_m - logged_x, delta_y_m=y_m - logged_y,
                    delta_norm_m=math.hypot(x_m - logged_x, y_m - logged_y),
                ))
    if not rows:
        raise ValueError("No RTK-fixed records")
    baseline_east, baseline_north = enu(reference["origin_lat_deg"], reference["origin_lon_deg"],
                                        reference["q4_lat_deg"], reference["q4_lon_deg"])
    norms = [row["delta_norm_m"] for row in rows]
    summary = dict(
        rows=len(rows), reference_baseline_m=math.hypot(baseline_east, baseline_north),
        x_mae_m=sum(abs(value) for value in dx) / len(dx),
        y_mae_m=sum(abs(value) for value in dy) / len(dy),
        coordinate_mae_m=sum(norms) / len(norms), coordinate_max_m=max(norms),
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with SUMMARY.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
