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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UAV_DIRECTORY = ROOT / "data" / "uav-20260903"
OUTPUT = ROOT / "analysis" / "csv" / "uav_local_frame_recalculation.csv"
SUMMARY = ROOT / "analysis" / "csv" / "uav_local_frame_recalculation_summary.csv"
WGS84_A_M = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def ecef(latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
    """WGS84 ellipsoidal surface point from geodetic latitude/longitude."""
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    radius = WGS84_A_M / math.sqrt(1 - WGS84_E2 * math.sin(latitude) ** 2)
    return (
        radius * math.cos(latitude) * math.cos(longitude),
        radius * math.cos(latitude) * math.sin(longitude),
        radius * (1 - WGS84_E2) * math.sin(latitude),
    )


def enu(origin_lat_deg: float, origin_lon_deg: float, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Horizontal east/north displacement in the tangent plane at O."""
    ox, oy, oz = ecef(origin_lat_deg, origin_lon_deg)
    px, py, pz = ecef(latitude_deg, longitude_deg)
    dx, dy, dz = px - ox, py - oy, pz - oz
    latitude, longitude = math.radians(origin_lat_deg), math.radians(origin_lon_deg)
    east = -math.sin(longitude) * dx + math.cos(longitude) * dy
    north = (-math.sin(latitude) * math.cos(longitude) * dx
             - math.sin(latitude) * math.sin(longitude) * dy
             + math.cos(latitude) * dz)
    return east, north


def local_xy(reference: dict[str, float], latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Return x/y with +y along the RTK O->Q4 reference baseline."""
    east, north = enu(reference["origin_lat_deg"], reference["origin_lon_deg"], latitude_deg, longitude_deg)
    q4_east, q4_north = enu(reference["origin_lat_deg"], reference["origin_lon_deg"],
                             reference["q4_lat_deg"], reference["q4_lon_deg"])
    length = math.hypot(q4_east, q4_north)
    if length < .01:
        raise ValueError("O and Q4 are too close")
    y_east, y_north = q4_east / length, q4_north / length
    return east * y_north - north * y_east, east * y_east + north * y_north


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
