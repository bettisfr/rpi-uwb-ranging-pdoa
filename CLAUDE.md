# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two loosely coupled halves:

1. **Field acquisition** (runs on a Raspberry Pi 5, `rpi5-01.local`, checkout at `/opt/pdoa-uwb-rpi`): reads a DWM1002 PDoA node over USB serial and logs ranging/PDoA samples from nine DWM1001/DWM1003 tags.
2. **Offline analysis** (`analysis/`, runs on a workstation): reproducible calibration benchmarks on the static campaign, then zero-shot transfer to an RTK-referenced UAV campaign.

`README.md` documents deployment, hardware, CSV fields, and the measurement campaign. `analysis/README.md` documents every benchmark command and its outputs in detail — read it before changing analysis code.

## Commands

Acquisition (C11, no dependencies beyond libc):

```bash
make                      # builds ./pdoa-monitor from src/pdoa_monitor.c
./pdoa-monitor -d /dev/ttyACM0 [--stream]
make web                  # runs scripts/pdoa-web.py (spawns pdoa-monitor)
./scripts/sync-rpi.sh [--web]   # rsync to Pi, build remotely, optionally restart pdoa-web.service
```

Tests use stdlib `unittest` (no pytest config); run from repo root:

```bash
python -m unittest discover -s tests                 # web server tests (POSIX only: os.killpg, /dev/null)
python -m unittest discover -s analysis/tests        # analysis tests
python -m unittest analysis/tests/test_splits.py     # single file
python -m unittest discover -s analysis/tests -k test_whole_conditions   # single test
```

Analysis (deps in `analysis/requirements.txt`; CUDA-only models need `analysis/requirements-gpu.txt`). Canonical order:

```bash
python analysis/src/main.py --build-static-dataset   # data/*/runs/*.csv -> analysis/csv/static_measurements.csv
python analysis/src/workflow.py --split-static        # audit + grouped train/val/test split
python analysis/src/workflow.py --compare-distance --lodo-distance
python analysis/src/workflow.py --lodo-phase-stratified --lobo-phase-continuous
python analysis/src/workflow.py --evaluate-generic-position
python analysis/src/uav_workflow.py --all             # requires static split + range model first
```

`python analysis/src/workflow.py --help` lists all actions.

## Acquisition architecture

- `src/pdoa_monitor.c` parses firmware TWR JSON lines from serial, maps short addresses to aliases via `config/tags.json`, writes `logs/pdoa_YYYYMMDD_HHMMSS.csv`, and reconnects every 5 s if the device disappears.
- `scripts/pdoa-web.py` is a single-file stdlib HTTP server (`ThreadingHTTPServer`) with inline HTML/JS. It launches `pdoa-monitor` as a child process and **tails the newest `logs/pdoa_*.csv`** rather than reading serial itself. Guided runs are written to `datasets/<experiment-id>/{experiment.json,runs/*.csv}`. It also has a `/drone` page that drives an external DJI telemetry binary (`/opt/dji-rpi-payload/...`) and aligns RTK with UWB samples.
- Campaign constants (`TAG_LAYOUT` bearings, `DISTANCES_M`, `ROTATIONS_DEG`, `RAW_FIELDS`) live at the top of `pdoa-web.py`; keep them consistent with `README.md` and the analysis audit.
- `tests/test_pdoa_web.py` loads the hyphenated script via `importlib` and stubs the monitor.
- Samples with `(x_cm, y_cm) == (0, 0)` are invalid and don't count toward run completion.
- Planned tag bearing is experiment metadata; firmware reports only raw PDoA and `Xcm`/`Ycm`. Don't conflate them.

## Analysis architecture

- `analysis/src/workflow.py` and `analysis/src/uav_workflow.py` are thin CLI dispatchers; logic belongs in `analysis/src/workflow/` (static) and `analysis/src/uav/` (UAV geometry). Keep the two packages independent.
- Scripts import siblings by bare name (`from workflow.splits import ...`), so they're run as `python analysis/src/<script>.py`; tests prepend `analysis/src` to `sys.path`.
- `workflow/contracts.py` fixes exported column contracts; `configs/*.json` version the split seed/fractions and model hyperparameters.
- Every run writes CSV/JSON artifacts with a manifest (source SHA-256, split, versions, status) to `analysis/outputs/`, `analysis/csv/`, `analysis/plots/` — all git-ignored. Plots read those artifacts, never in-memory models. A manifest with status `running` or `complete_with_partial_fits` is not a valid result.
- Where to modify what (from `analysis/README.md`): range corrections in `workflow/distance_models.py`, PDoA corrections in `workflow/phase_*.py`, position reconstruction in `workflow/position_benchmark.py`, UAV geometry in `uav/config.py`/`uav/coordinates.py`. Keep `data.py`, `splits.py`, `metrics.py` fixed when comparing models.

## Methodological rules (enforced by design, keep them)

- Ground-truth distance/bearing/position are targets only — never input features.
- Fit preprocessing and models on train only; validation for comparison; test subset is reserved and not evaluated by current commands. LODO/LOBO folds use the development pool (train + val).
- Splits group whole physical conditions (height, distance, rotation, all tags/repeats) — never split individual samples.
- Circular metrics/wrapping (`[-180, 180)`) for phase residuals; linear metrics for range/position.
- CUDA models fail if CUDA is unavailable; never silently fall back to CPU.
- Pre-register model, protocol and success criterion in `analysis/README.md` *before* running a new benchmark; record deviations there.
- `workflow.py --final-test` evaluates the grouped test once and then refuses to rerun; do not use `--force-final-test` except after a crash.
- Intervals use `workflow/bootstrap.py` (resample configurations, never single samples; UAV uses flight × tag × 10 s blocks).
- Firmware `x_cm`/`y_cm` embed a ~93.5° PDoA offset (T0.1); never use them as an independent bearing estimate.
- UAV logs hold the 1 Hz RTK fix across ~10 Hz rows: interpolate only between fix changes (`evaluate_uav_range.fix_track`/`track_at`) with the cross-flight latency. PDoA mount (offset, sign) is only identifiable with the sign fixed to the static convention.
- Do not tune `uav/config.py` from UAV results. `*_ad_hoc.py` and `evaluate_uav_geometry_residual.py` use flight ground truth and are diagnostics only — keep them out of `uav_workflow.py --all`.
