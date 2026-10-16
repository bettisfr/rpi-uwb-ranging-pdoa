"""Fixed, surveyed configuration of the UAV campaign.

Keep campaign constants here so range, phase, and position diagnostics use
the same physical convention.  Changing a value requires regenerating every
UAV result that depends on it.
"""

from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis"
UAV_DATA = ROOT / "data" / "uav-20260903"
UAV_REFERENCE = UAV_DATA / "reference.json"
UAV_PREDICTIONS = ANALYSIS / "outputs" / "uav_range_transfer" / "predictions.csv"

# Heights are measured from the local terrain reference used by the flight
# controller.  The offsets convert them to the node--tag phase-center gap.
NODE_ANTENNA_OFFSET_M = 0.55
TAG_HEIGHT_M = 0.45

MAX_ALIGNMENT_AGE_S = 0.10
# The logger writes ~10 Hz rows but RTK position refreshes at ~1 Hz and is held
# in between; interpolate between fix changes, never across a missed update.
RTK_MAX_FIX_GAP_S = 1.50

# (Bearing from the positive local y-axis in degrees, radial distance in m).
TAG_LAYOUT: dict[str, tuple[float, float]] = {
    "dw00": (15.0, 28.0), "dw01": (35.0, 18.0), "dw02": (55.0, 10.0),
    "dw03": (75.0, 24.0), "dw04": (90.0, 30.0), "dw05": (105.0, 24.0),
    "dw06": (125.0, 10.0), "dw07": (145.0, 18.0), "dw08": (165.0, 28.0),
}


def tag_xy(tag: str) -> tuple[float, float]:
    """Return the surveyed local coordinate of a fixed UAV-campaign tag."""
    bearing_deg, radius_m = TAG_LAYOUT[tag]
    bearing_rad = math.radians(bearing_deg)
    return radius_m * math.cos(bearing_rad), radius_m * math.sin(bearing_rad)
