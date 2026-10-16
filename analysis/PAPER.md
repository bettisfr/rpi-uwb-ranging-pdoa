# Reproducing the paper

This is the canonical workflow for the current manuscript. The former
`analysis/test-plan` implementation has been integrated into the source tree;
use the current branch, including the added parametric-test and LOHO scripts.
Run commands from the repository root. No paths depend on the original author’s
home directory. Raw measurements remain under `data/`; normalized data,
predictions, metrics, plots, fitted models, and manifests are generated locally.

## Environment and complete run

Use Python 3.11 or newer in a virtual environment:

```bash
python -m pip install -r analysis/requirements.txt
python -m pip install -r analysis/requirements-gpu.txt
python analysis/src/check_gpu.py
python analysis/src/reproduce_paper.py --all --dry-run
python analysis/src/reproduce_paper.py --all --resume
```

The full run includes six distance models in grouped validation, LODO, and
LOHO. Their scikit-learn fits run on CPU; SVR and repeated MLP fits are the
slowest stages. The continuous LOBO phase MLP requires CUDA and does not
silently fall back to CPU. Its hardware backend does not change its model
name in the manuscript. Individual phase-SVR experiments are historical
diagnostics and are not required to regenerate the paper tables.

`--dry-run` makes no changes. `--resume` reuses only stages completed by this
runner with matching code/data fingerprints and existing products. Full
reproduction otherwise fits the documented models again. It does not tune
hyperparameters on evaluation subsets. A previously completed frozen-model
evaluation is reused after checking the source hash and split; historical
results are not relabeled as a newly untouched test.

One command log per stage is saved in `analysis/outputs/paper_pipeline_logs/`.
`analysis/outputs/paper_pipeline_manifest.json` records interpreter/package
versions, input/code fingerprint, completion status, and generated products.
Generated directories are ignored by Git; commit source/config changes, not
the large generated prediction files.

## Rerun a specific part

The runner automatically includes prerequisites. Examples:

```bash
python analysis/src/reproduce_paper.py --stages final-models --resume
python analysis/src/reproduce_paper.py --stages static-windows --resume
python analysis/src/reproduce_paper.py --stages uav --resume
```

`final-models` needs normalization/splitting but does not require the slow
distance-model comparison. `uav` adds those frozen models, RTK processing,
cross-flight mounting alignment, and window evaluation. `static-windows`
needs the LODO range and phase predictions; a clean run therefore includes
the LODO benchmark. Use `--dry-run` before launching a stage to inspect
exactly what will run.

## Table and figure sources

The final export command is read-only with respect to models and source data:

```bash
python analysis/src/export_paper_tables.py
```

It fails with the missing filename if a prerequisite has not been generated.
It never substitutes values copied manually from the manuscript. Exported
CSV files retain numerical precision; round values only when rendering a
table or plot. `paper_tables/manifest.json` hashes the analysis CSV sources.

| Manuscript label | Export in `outputs/paper_tables/` | Responsible stage/source |
| --- | --- | --- |
| `tab:static-dataset` | `static_dataset.csv` | `data`; normalized static records |
| `tab:static-tag-summary` | `static_tag_summary.csv` | `data`; per-tag runs, cumulative time, within-configuration variability |
| `tab:uav-tag-layout` | `uav_tag_layout.csv` | `uav/config.py` and raw aligned flight rows |
| `tab:uav-flight-summary` | `uav_flight_summary.csv` | DJI KML horizontal path, decoded DJI duration/speed, aligned height samples |
| `tab:range-models` | `range_models.csv` | `range-validation`, `range-lodo`, `range-loho`, `final-models` |
| `tab:pdoa-models` | `pdoa_models.csv` | `phase-lodo`, `phase-calibration`, `phase-mlp`, `final-models` |
| `tab:position-models` | `position_models.csv` | `static-position`, `final-models` |
| `tab:window-position` | `window_position.csv` | `static-windows` at K=10, 20, 30 |
| `tab:uav-range-transfer` / `tab:uav-pdoa-transfer` | `uav_transfer.csv` | `uav`; includes single-sample errors on exactly the windows being averaged |
| `fig:static-range-pdoa` | `analysis/csv/static_range_pdoa_h{0,2,4}.csv` | `data`; inputs for the standalone plot in the paper |
| UAV trajectory plot | `generate_uav_trajectory_data.py` | Plot-only one-second trajectory averaging; not evaluation resampling |

The notation table and construction figures are manuscript definitions,
not model outputs. The exported model-table files are in long format and
carry protocol/model identifiers, units, sample counts, and source paths.
They include full computed results; the manuscript may display a subset
(for example, its distance Test columns show the three parametric models).

## Fixed evaluation contracts

- Static split: whole height–distance–rotation runs, 60/20/20, seed 42,
  stratified by height. Repeated observations of one run stay together.
- Range models use measured horizontal distance only. Known height is used
  for projection. Seven undefined projections and 20 gross horizontal-range
  errors above 3 m are disabled using the documented static quality mask.
  The original data remain available. Phase fitting retains finite phase
  observations without applying the distance mask.
- Grouped validation fits on train; LODO/LOHO/LOBO fit only on the remaining
  development-pool conditions. The grouped test is excluded from these fits.
- Final range comparison and final static position models fit on
  train+validation. The chosen primary position chain is quadratic horizontal
  range plus two-harmonic phase correction; model forms and thresholds remain
  fixed. Earlier exploratory exposure of the grouped test is acknowledged.
- The physical phase characteristic is initially described at low elevation;
  its benchmark version is refitted within each training fold. Tag-specific
  harmonic and constant models are diagnostic: tag identity encodes the fixed
  bearing. Their LOHO fit excludes an entire height; LOBO is inapplicable to a
  model that requires training data from the excluded tag.
- Static windows average corrected horizontal range arithmetically and phase
  circularly, then reconstruct position. Each uses LODO predictions.
- RTK uses fixed status 50. Position is interpolated between actual position
  changes, with a 1.5 s maximum bracket. Height uses separate change points.
  Each flight uses the latency estimated on the other flight. Hardware
  geometry/constants are centralized in `src/uav/config.py`.
- The UAV primary chain uses the same static quadratic and global-harmonic
  models. Mount yaw is fitted on one flight and applied to the other. Raw
  phase and the physical characteristic remain separately labeled diagnostics.
- UAV windows first reconstruct positions in the fixed test-bed frame, then
  average 20 positions. Windows with sample gaps above 0.5 s are excluded.
  Raw-phase circular standard deviation <=20 degrees selects stable windows.
  Coverage is relative to gap-free windows. Single-sample errors on the same
  windows isolate the benefit of averaging from sample selection.
- Bootstrap intervals resample whole tag–height–distance–rotation
  configurations for static data and tag-specific ten-second flight blocks
  for UAV data (1000 replicates, seed 42).

## Improve a model without changing the experiment

Edit range estimators in `src/workflow/distance_models.py` and phase estimators
in `src/workflow/phase_*.py`. Keep data, splits, quality policy, geometry,
metrics, and exports fixed. Add new model names rather than silently changing
the meaning of an existing CSV row. Fit preprocessing inside each training
fold. Ground-truth distance, bearing, and coordinates are targets or analysis
variables, never generic model inputs. Flight-specific tuning belongs in a
separate diagnostic, not in the frozen transfer chain.

## Checks

```bash
python -m unittest discover -s analysis/tests -v
```

The tests cover split isolation, distance eligibility, circular statistics,
reconstruction, bootstrap grouping, and RTK change-point interpolation.
The full numerical run remains the integration check for the paper tables.
