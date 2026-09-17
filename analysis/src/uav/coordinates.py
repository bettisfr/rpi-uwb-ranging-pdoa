"""WGS84/ENU/local-frame conversions shared by UAV analyses."""

from __future__ import annotations

import math


WGS84_A_M = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def ecef(latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
    """Return the WGS84 ellipsoidal surface point for geodetic coordinates."""
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    radius = WGS84_A_M / math.sqrt(1 - WGS84_E2 * math.sin(latitude) ** 2)
    return (
        radius * math.cos(latitude) * math.cos(longitude),
        radius * math.cos(latitude) * math.sin(longitude),
        radius * (1 - WGS84_E2) * math.sin(latitude),
    )


def enu(origin_lat_deg: float, origin_lon_deg: float, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Return horizontal east/north displacement in the tangent plane at O."""
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
    """Return x/y with positive y along the surveyed O-to-Q4 baseline."""
    east, north = enu(reference["origin_lat_deg"], reference["origin_lon_deg"], latitude_deg, longitude_deg)
    q4_east, q4_north = enu(
        reference["origin_lat_deg"], reference["origin_lon_deg"],
        reference["q4_lat_deg"], reference["q4_lon_deg"],
    )
    length = math.hypot(q4_east, q4_north)
    if length < 0.01:
        raise ValueError("O and Q4 are too close")
    y_east, y_north = q4_east / length, q4_north / length
    return east * y_north - north * y_east, east * y_east + north * y_north


def q4_heading_deg(reference: dict[str, float]) -> float:
    """Return the O-to-Q4 heading clockwise from ENU north."""
    east, north = enu(
        reference["origin_lat_deg"], reference["origin_lon_deg"],
        reference["q4_lat_deg"], reference["q4_lon_deg"],
    )
    return math.degrees(math.atan2(east, north))
