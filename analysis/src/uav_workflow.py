#!/usr/bin/env python3
"""Run the reproducible RTK-referenced UAV analysis in dependency order.

The static split and quadratic range model must be available first.  This
entry point intentionally runs only the documented transfer analysis; scripts
named ``*_ad_hoc`` or ``*_residual`` remain optional diagnostics because they
fit or inspect flight-specific quantities.
"""

from __future__ import annotations

import argparse

from audit_uav import main as audit
from evaluate_uav_bearing_position import main as bearing_position
from evaluate_uav_pdoa import main as pdoa_diagnostic
from evaluate_uav_range import main as range_transfer
from plot_uav_range_transfer import main as range_plot
from recalculate_uav_local_frame import main as recalculate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="audit aligned UWB/RTK source logs")
    parser.add_argument("--frame", action="store_true", help="recalculate local RTK coordinates")
    parser.add_argument("--range", action="store_true", help="evaluate frozen static range correction")
    parser.add_argument("--pdoa", action="store_true", help="run the oracle-mount PDoA diagnostic")
    parser.add_argument("--bearing", action="store_true", help="evaluate cross-flight bearing and position")
    parser.add_argument("--plot-range", action="store_true", help="plot range-transfer residuals")
    parser.add_argument("--all", action="store_true", help="run audit, frame, range, PDoA, bearing, and range plot")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("choose an action; see --help")

    if args.all or args.audit:
        audit()
    if args.all or args.frame:
        recalculate_frame()
    if args.all or args.range:
        range_transfer()
    if args.all or args.pdoa:
        pdoa_diagnostic()
    if args.all or args.bearing:
        bearing_position()
    if args.all or args.plot_range:
        range_plot()


if __name__ == "__main__":
    main()
