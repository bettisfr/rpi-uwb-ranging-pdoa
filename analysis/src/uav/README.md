# UAV analysis package

This package centralizes the physical quantities shared by the dynamic
campaign.  It is deliberately independent of the static learning code in
`analysis/src/workflow/`.

| File | Responsibility | Change it when... |
| --- | --- | --- |
| `config.py` | surveyed tag layout, antenna-height offsets, log paths, alignment thresholds | the field layout, payload, or quality policy changes |
| `coordinates.py` | WGS84 → ENU → local O/Q4 frame | the reference-frame definition changes |
| `angles.py` | angle wrapping and circular averages | a circular convention changes |

Do not tune `config.py` from UAV test results.  These values describe the
campaign geometry and are inputs to every range, phase, and position result.
Flight-specific or ground-truth-conditioned experiments remain explicitly
separate in `evaluate_uav_range_ad_hoc.py` and
`evaluate_uav_geometry_residual.py`.
