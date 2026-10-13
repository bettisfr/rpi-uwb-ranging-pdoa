# Measurement-analysis workflow

This directory keeps reproducible analysis separate from field acquisition and
from paper-specific plotting code.  It has two deliberately separate parts:
the controlled static calibration workflow in `src/workflow/`, and the
RTK-referenced UAV transfer workflow in `src/uav/`.

## Where to modify a model

Keep data loading, surveyed geometry, splits, metrics, and exports fixed when
comparing a new correction.  Add or alter a model only in the indicated area:

| Question | Modify | Keep fixed |
| --- | --- | --- |
| Improve static range correction | `src/workflow/distance_models.py` | `data.py`, `splits.py`, `metrics.py` |
| Improve static PDoA correction | `src/workflow/phase_*.py` | circular metrics and the selected split |
| Change derived static position reconstruction | `src/workflow/position_benchmark.py` | range/PDoA input contracts |
| Change UAV geometry or reference frame | `src/uav/config.py`, `src/uav/coordinates.py` | do not tune from dynamic-test results |
| Add an exploratory UAV diagnostic | a new `evaluate_uav_*.py` script | label it as diagnostic if it uses flight ground truth |

`src/workflow.py` and `src/uav_workflow.py` are command-line entry points;
they should remain thin dispatchers.  Generated `csv/`, `outputs/`, and
`plots/` files are ignored by Git.  Every substantive run writes CSV/JSON
artifacts with provenance so a plot or paper table never depends on in-memory
model objects.

## Reproducing the documented analyses

Run commands from the repository root.  On this workstation, the maintained
interpreter is `/home/fra/pyvenv/bin/python`; throughout this document,
`python` denotes that interpreter (or an equivalent environment created by
another user).  Confirm it before running a benchmark:

```bash
/home/fra/pyvenv/bin/python --version
```

Install the regular dependencies first.  The continuous LOBO MLP and the
separate PyTorch comparison require the CUDA dependency file as well; they do
not silently fall back to CPU.

```bash
/home/fra/pyvenv/bin/python -m pip install -r analysis/requirements.txt
/home/fra/pyvenv/bin/python -m pip install -r analysis/requirements-gpu.txt

# Normalize, audit, and create the static grouped split.
python analysis/src/main.py --build-static-dataset
python analysis/src/workflow.py --split-static

# Static range, PDoA/bearing, and derived position benchmarks.
# The LOBO command uses CUDA for its MLP fold in addition to CPU baselines.
python analysis/src/workflow.py --compare-distance --lodo-distance
python analysis/src/workflow.py --lodo-phase-stratified --lobo-phase-continuous
python analysis/src/workflow.py --evaluate-generic-position
python analysis/src/windowed_position_sensitivity.py

# UAV: only after the static split and range configuration above are available.
python analysis/src/uav_workflow.py --all
```

On the local machine, replace every `python` command below with
`/home/fra/pyvenv/bin/python` if that interpreter is not already active.

The UAV runner performs the source audit, independent RTK-frame recalculation,
frozen static-range transfer, PDoA diagnostic, cross-flight bearing/position
evaluation, and the range-transfer plot in dependency order.  It does not run
the optional `*_ad_hoc.py` experiments, because those use flight-specific
ground truth and must not be confused with deployable transfer results.

## Layout

```text
analysis/
  configs/       versioned benchmark definitions
  csv/           ignored normalized datasets and descriptive summaries
  outputs/       ignored benchmark plans, predictions, metrics, and models
  plots/         ignored exploratory matplotlib figures
  src/
    main.py      current static-data normalization and descriptive analysis
    workflow/    benchmark contracts, splits, models, metrics, and exports
    workflow.py  command-line entry point for the benchmark workflow
    uav/         surveyed UAV configuration, coordinates, and angle utilities
    uav_workflow.py  command-line entry point for the documented UAV transfer workflow
```

Generated files are intentionally ignored by Git.
Each benchmark run writes a small JSON manifest and CSV files, so plots and
paper tables can be regenerated from a stable tabular interface rather than
from model-specific objects.

## Benchmark sequence

The static data are the training and controlled-validation source.
The UAV data remain a held-out dynamic transfer campaign.

1. Establish a coplanar baseline using only runtime-observable quantities.
2. Add observable enrichment, such as clock offset and cyclic encodings of
   raw PDoA phase and tag rotation.
3. Train a single height-aware model on the joint static dataset.
4. Evaluate grouped configurations, leave-one-distance-out, and
   leave-one-height-out splits.
5. Freeze the selected static model for zero-shot UAV evaluation.
   Fine-tuning, if studied, is an explicit adaptation ablation and must use a
   disjoint time/flight split.

Ground-truth distance, bearing, and position are targets or analysis variables;
they must never be input features for a predictive benchmark.

## Targets and metrics

The first models predict corrections, not coordinates directly:

- `ground_distance_error_m` for the range correction;
- `eta_residual_deg` for the PDoA-phase correction;
- derived node-local position error after applying the two corrections.

CSV metric rows include bias, MAE, RMSE, standard deviation, and `r2` where it
is meaningful for a signed scalar target.
Circular metrics are reserved for phase residuals.
MAE and RMSE remain the primary summaries for nonnegative position error.

## Use

From the repository root:

```bash
python analysis/src/workflow.py --prepare --plan
```

This command is safe: it creates only ignored, header-only output files and a
benchmark plan/manifest.
The exported column contracts are already fixed in `workflow/contracts.py`.

## First step: static-data audit

```bash
python analysis/src/workflow.py --audit-static
```

Requires the existing `csv/static_measurements.csv`, produced by
`src/main.py --build-static-dataset` when needed.
Exports `outputs/static_configuration_inventory.csv` (all 1620 planned
height/distance/rotation/tag configurations, including missing ones),
`outputs/static_run_inventory.csv` (observed run/tag counts), and
`outputs/static_audit.json` (counts, source SHA-256, errors, and scope).
The audit checks required numeric fields, the fixed tag/bearing mapping,
run-factor consistency, and horizontal-range targets against Pythagoras.
It reports invalid horizontal projections separately and does not silently
remove samples. Validation errors cause a nonzero exit after writing the report.
Run IDs include the experiment ID because run filenames repeat across heights.
The audit covers the normalized CSV; checking raw-source completeness and
duplicate acquisition events remains a later step.

## Grouped training/validation/test split

```bash
python analysis/src/workflow.py --split-static
```

Runs the static audit, then assigns physical conditions to train (60%),
validation (20%), and test (20%), separately within each height.
The seed (42) and fractions are versioned in `configs/static_benchmark.json`.
Each group contains all tags and all repetitions at the same height, distance,
and rotation, including repeated runs with different filenames.
Fractions apply to groups; the actual numbers of samples vary with missing tags.
Coplanar analyses select h=0 from the same assignments; the joint analysis uses
all three heights. The grouped split evaluates unseen combinations and does
not guarantee unseen distances or heights; those require the later holdout tests.

Exports in `outputs/`:

- `static_grouped_assignments.csv`: one row per observed run/tag pair, with
  group, subset, and counts. Join to measurements using experiment_id, run_file,
  and tag (many measurement rows to one assignment).
- `static_grouped_summary.csv`: actual group/run/sample counts overall and by height.
- `static_grouped_manifest.json`: dataset hash, grouping rule, seed, and fractions.

All normalized observations retain an assignment. The seven currently invalid
horizontal projections will require target-specific eligibility filtering when
training models; they do not cause reassignment of any group.
For range calibration only, the source is also kept intact while the 20 valid
observations whose absolute error against the surveyed test-bed geometry exceeds
3 m are marked disabled. This conservative quality mask removes isolated gross
glitches from fitting and metrics; it is recorded in every distance manifest.
Fit preprocessing and models on train only, tune on validation, and use test
only for final evaluation. Generated split files are authoritative for that
dataset hash and must be retained with the benchmark results.

## Constant horizontal-distance correction

```bash
python analysis/src/workflow.py --fit-offset
```

Audits the data and regenerates the deterministic grouped split, then fits one
sample-weighted mean horizontal-distance error using eligible training samples
pooled across all heights. Subtracting this offset from the projected distance
produces the corrected estimate. No tuning uses validation or test measurements.
This first baseline corrects horizontal distance after projection, not 3D range.

`outputs/constant_offset/` contains `model.json` (coefficient and provenance),
`predictions.csv` (eligible observations, original CSV line IDs, groups and
subsets), and `metrics.csv` (raw/corrected bias, MAE, RMSE, population standard
deviation, and distance R2 overall and per height for each subset).
Undefined projections and the disabled gross glitches are excluded from this
target only and their counts are saved in the model JSON. R2 here evaluates distance estimates, not
prediction of residuals, and is empty for constant ground-truth distances.
Repeating this command regenerates the baseline outputs in the same directory.

## Linear horizontal-distance correction

```bash
python analysis/src/workflow.py --fit-linear
```

Requires the existing grouped split and verifies its dataset hash. Fits ordinary
least squares `rho_corrected = a * rho_measured + b` on eligible training rows,
pooling all heights, and compares raw, offset and linear estimates on training
and validation only. Test metrics are not computed by this command.
The source is never modified; the shared distance-quality mask is applied.

`outputs/linear_regression/` contains model coefficients and provenance in
`model.json`, per-observation `predictions.csv`, aggregate/per-height
`metrics.csv`, and `validation_by_configuration.csv` grouped by height,
distance and tag (available rotations pooled). Additional diagnostics are the
nearest-rank 90th percentile and maximum absolute error. Distances and errors
are exported in meters. These files are regenerated on each invocation.

## Quadratic comparison

```bash
python analysis/src/workflow.py --fit-quadratic
```

Fits a degree-two polynomial to horizontal distance using the same training
samples and existing grouped split. Training-only centering/scaling and NumPy
least squares improve numerical conditioning. The shared distance-quality mask
is applied; undefined horizontal projections remain excluded as before.
Exports all four methods (raw, offset, linear, quadratic) under
`outputs/quadratic_regression/`, using the same CSV layout as the linear run.
The JSON records coefficients, normalization, the linear/offset baselines and
data/split provenance. Evaluation remains on training and validation; the test
set is reserved. The 90th percentile uses the nearest-rank convention.

## Random Forest, one observed feature

```bash
python analysis/src/workflow.py --fit-forest
```

Fits a RandomForestRegressor using only measured horizontal distance, with
100 trees, maximum depth 12, minimum 20 samples per leaf, seed 42 and two workers.
These parameters are fixed before evaluation; no validation tuning is performed.
The target is true horizontal distance, as for the linear/quadratic fits.
All distance-quality-enabled training samples are retained.

Exports all five methods under `outputs/random_forest/`, with the same
predictions/metrics/validation diagnostics as previous comparisons.
`model.json` records parameters, versions and provenance; `model.joblib` stores
the fitted forest for reuse (load only trusted local artifacts).
Evaluation uses training and validation; test is not evaluated.
Nominal distances repeat across this grouped split, so strong performance
may reflect learning the 2 m distance grid. Generalization to unseen distances
must be assessed separately with leave-one-distance-out.

## Complete distance comparison: SVR and MLP

```bash
python analysis/src/workflow.py --compare-distance
python analysis/src/workflow.py --lodo-distance
```

Both commands consume the existing grouped assignments and verify the dataset
hash. The first compares raw measurements, offset, linear, quadratic, Random
Forest, RBF SVR and MLP on the original validation subset. Models fit only the
original training subset. The second runs 15 folds over the **development pool
(original training + validation)**, excluding every observation at the held-out
distance from fitting, across all heights, tags and rotations. The original
grouped test subset stays excluded from both protocols. Each eligible
development observation is evaluated once in leave-one-distance-out.
The endpoint folds (2 m and 30 m) are extrapolation tests; the others test
interpolation across a missing distance. No hyperparameter tuning uses these
held-out observations.

Parameters are fixed in `configs/distance_models.json`. All six fitted models
use the same single observed feature and eligible samples. Input scaling for
SVR and MLP, and target scaling for MLP, are fitted on training data within each
fold. MLP has two hidden layers (32, 16); its stopping criterion uses training
loss, with no internal random validation split. Iterations, convergence warnings
and training loss are recorded. SVR uses C=10, epsilon=0.05 m, and gamma=1 after
input standardization.

To accelerate exact kernel SVR, identical training (input, target) pairs are
represented once with their multiplicity as sample weight. This preserves the
SVR objective for every enabled row; no rounding or subsampling is used. A test
compares this implementation with uncompressed SVR on repeated observations.

Outputs are in `outputs/distance_comparison/` and `outputs/distance_lodo/`:

- `manifest.json`: parameters, source hash, split, versions, fit times, warnings
  and completion status; never treat a `running` manifest as a complete run.
- `assignments.csv`: original CSV line IDs and training/evaluation roles per fold.
- `metrics.csv`: per-fold and per-height metrics.
- `metrics_by_distance.csv`: residual summaries at individual distances.
- `overall_metrics.csv`: sample-weighted pooled evaluation metrics, not the
  arithmetic mean of fold metrics. R2 for a single-distance fold is undefined.
- Per-fold directories: fitted `.joblib` estimators and **wide**
  `predictions.csv`, one evaluation row with a prediction column per method.
  This avoids repeating sample metadata seven times. A line ID joins back to
  the normalized source identified by its SHA-256 hash.

All metrics use meters; standard deviation uses ddof=0 and P90 uses nearest
rank. The 3 m distance-quality mask is applied without output clipping; source
datasets are not altered. These commands do not alter the raw source datasets or the earlier result folders.
Rerunning a command regenerates its own result directory's named artifacts.

```bash
python analysis/src/plot_distance_lodo.py
```

Exports LODO range plots: a compact overall comparison separates simple
calibration candidates from nonlinear baselines, while a second plot reports
MAE at each held-out distance.

## PDoA diagnostics and calibration

### Raw PDoA diagnostics by tag, height, and rotation

```bash
python analysis/src/plot_pdoa_diagnostics.py
```

Exports raw, circular PDoA-residual summaries without applying a correction.
`csv/pdoa_raw_by_tag_height.csv` separates h0, h2, and h4 for each tag; its
companion PDF makes the deployment-bearing pattern visible. The rotation CSV
and PDF pool the three heights and instead compare the four known tag rotations.
These diagnostics keep every finite phase record and are deliberately separate
from range-quality filtering.

### Stratified calibration at unseen distances

```bash
python analysis/src/workflow.py --lodo-phase-stratified
```

Evaluates circular-offset and two-harmonic trigonometric corrections with each
of four known deployment conditions: global, tag/bearing, tag rotation, and the
joint tag/bearing--rotation condition. Each of the 15 nominal distances is held
out in turn across every height and rotation. In this fixed test-bed, tag ID and
bearing are equivalent labels; neither ground-truth distance nor position is an
input feature. Outputs are in `outputs/pdoa_stratified_lodo/`.

```bash
python analysis/src/plot_phase_stratified.py
```

Exports overall MAE/RMSE and tag-level MAE PDF plots, with the latter directly
contrasting raw, global-trigonometric, bearing-trigonometric, and joint models.

## Generic position reconstruction at unseen distances

```bash
python analysis/src/workflow.py --evaluate-generic-position
```

Combines matching out-of-fold range and PDoA predictions to compare raw firmware
coordinates with geometric reconstructions. The generic chain uses only range,
PDoA, and known height difference. Tag, bearing, rotation, and ground-truth
coordinates never select a correction. The positive-y half-plane is a known
constraint of the static test-bed; infeasible PDoA projections are explicitly
clipped to the physical disk and their rate is reported.

```bash
python analysis/src/plot_generic_position.py
```

Exports overall and per-tag PDF comparisons of firmware coordinates and the
generic geometric reconstructions.

## Continuous generic PDoA correction at an unseen bearing

```bash
python analysis/src/workflow.py --lobo-phase-continuous
```

Holds out one complete tag/bearing at a time. The continuous models receive
only measured PDoA, measured range, and known height difference; tag, bearing,
and rotation are deliberately excluded. The fixed harmonic model and CUDA MLP
both predict the expected PDoA phase as a continuous quantity. The MLP keeps
each training fold and its minibatch permutation on CUDA; its manifest records
the device, runtime, training loss, and elapsed time for all nine folds. If
CUDA is unavailable the command fails rather than substituting a CPU result.

```bash
python analysis/src/plot_phase_continuous.py
```

Exports overall and per-held-bearing PDF comparisons for the continuous LOBO benchmark.

## Optional sector-classifier position diagnostic

```bash
python analysis/src/workflow.py --evaluate-sector-position
```

Learns a nine-sector classifier from observed UWB features and selects a bearing-specific PDoA expert without receiving the true sector at inference. It is evaluated LODO and exported separately because it is a fixed-sector diagnostic, not a continuous generic model.

## PDoA baseline and circular offset

```bash
python analysis/src/workflow.py --fit-phase-offset
```

Uses the same training/validation assignments and fits a single circular mean
of the training residual `wrap(pdoa_deg - eta_expected_deg)`, pooling heights.
Correction subtracts that offset and wraps into [-180,180). The reference is
the nominal half-wavelength far-field phase `-180 * gt_x_m / gt_range_3d_m`,
including height. This evaluates phase, not geometric bearing.
The fitted circular mean minimizes chordal loss, not angular MAE/RMSE directly.

All finite phase observations are retained, including those whose measured
range cannot be projected horizontally. Test is not evaluated. Files in
`outputs/pdoa_offset/` include predictions, aggregate/per-height metrics,
tag/height metrics and `model.json`. Metrics report circular bias, circular
standard deviation, resultant length, shortest-angle MAE/RMSE and nearest-rank
P90 in degrees. An undefined circular mean/standard deviation is exported as
empty; an undefined training offset fails fitting. Linear R2 is not used.

## Trigonometric PDoA correction

```bash
python analysis/src/workflow.py --fit-phase-trig
```

Fits sine and cosine of the circular residual using the fixed basis
`[1, sin(eta), cos(eta), sin(2 eta), cos(2 eta)]`, where eta is measured phase.
Two harmonics are fixed before evaluation. Training-only least squares fits
the two residual components; atan2 reconstructs the correction, which is
subtracted from measured phase and wrapped. No ground-truth quantity is an
input feature. All heights are pooled, with no outlier removal.

Exports predictions, raw/offset/trigonometric metrics by subset, height and tag,
and reusable JSON coefficients to `outputs/pdoa_trigonometric/`. Evaluation
uses the same training/validation split. For a predicted residual vector norm
below 1e-8, the model falls back to the training circular offset and records the
event. Vector norm is diagnostic only, not a calibrated confidence probability.

## First phase learning experiments

```bash
python analysis/src/workflow.py --compare-phase
```

Compares fixed RF and MLP models with (a) sine/cosine of measured phase only
and (b) those features plus measured 3D range, known height and sine/cosine of
known tag rotation. Targets are the two circular residual components, as in
the trigonometric model. Original training only is used to fit every model
and scaler; original validation is evaluated, with test untouched.
All finite observations remain, including measured ranges below height.
RF uses 100 trees, depth 12 and minimum 20 samples per leaf; MLP uses (32,16)
hidden layers and a maximum of 200 epochs, with training-loss stopping.
No parameter tuning on validation is performed. Parameters, warnings,
iterations and training times are saved alongside the fitted estimators.

`outputs/pdoa_learning/` contains wide validation predictions, summary metrics,
tag/height diagnostics, joblib models and a JSON manifest. Raw, circular-offset
and trigonometric baselines are recomputed on the identical evaluation rows.
Cross-distance/height validation remains a subsequent step; these results do
not establish generalization outside this grouped split.

## SVR phase trial

```bash
python analysis/src/workflow.py --fit-phase-svr
```

Uses the same two feature sets and circular-residual targets as RF/MLP.
Two RBF SVRs predict cosine and sine separately. Fixed parameters are C=1,
epsilon=0.05 in component units, gamma=scale, tol=0.001, cache=512 MB, and
100000 maximum iterations per component. All training samples retain their
weight; exact duplicate input/target pairs are compressed with multiplicity
weights. Input scaling is fitted on original training samples only.
Models are evaluated on the unchanged validation subset, with no outlier
filtering or test evaluation.

Outputs are saved separately in `outputs/pdoa_svr/`. Each component's iteration
count, convergence status and warnings are recorded in the manifest. A capped
fit may export provisional predictions; in that case the manifest status is
`complete_with_partial_fits`, and it must not be reported as a converged model.

## Optional GPU environment

The local machine has an NVIDIA RTX 3060 with 12 GB VRAM. PyTorch CUDA
dependencies are specified separately in `requirements-gpu.txt`:

```bash
python -m pip install -r analysis/requirements-gpu.txt
```

This prepares PyTorch for future GPU models; existing scikit-learn SVR/MLP
implementations still execute on CPU. Installing PyTorch does not switch their
backend. GPU experiments will require a separate implementation and the same
training/validation protocol.

Verify CUDA forward/backward operations and optimizer updates with:

```bash
python analysis/src/check_gpu.py
```

The synthetic check writes versions, device information and loss values to
`outputs/gpu_check.json`; it is not a dataset experiment or a paper result.

Once that check passes, run the GPU phase comparison with:

```bash
python analysis/src/workflow.py --fit-phase-torch
```

It trains two fixed PyTorch MLPs on CUDA: phase-only and enriched. Both predict
the cosine/sine PDoA residual using training-only standardization, two hidden
layers (64, 32), AdamW, 200 epochs, batch size 4096, and seed 42. Validation is
never used for early stopping or parameter selection. Artifacts, metrics and
provenance are written to `outputs/pdoa_torch_gpu/`.

## Phase 0 integrity diagnostics

```bash
python analysis/src/workflow.py --t0-firmware-frame
python analysis/src/workflow.py --t0-mlp-check        # CUDA required
python analysis/src/workflow.py --t0-bearing-label    # needs --lodo-phase-stratified
```

- `t0_1_firmware_frame/`: tests whether the firmware `x_cm`/`y_cm` error is a
  frame convention (rigid rotation/reflection/translation fitted on train) and
  fits the firmware phase-to-angle map
  `x/r = clip(sign * scale * wrap(pdoa - offset) / 180)` by grid search on train.
  The fitted sign, offset and scale are stored in the manifest.
- `t0_2_mlp_check/`: the continuous-feature MLP and harmonic models on the
  grouped split and on each leave-one-bearing-out fold, reporting a 20k-sample
  training-error probe next to the held-out error.  Low training error with
  high held-out error indicates extrapolation across discrete bearings, not a
  fitting bug.
- `t0_3_bearing_label/`: compares the per-bearing PDoA correction with a
  label-only baseline (per-tag circular mean of training targets, PDoA unused),
  using the folds of `pdoa_stratified_lodo`.

## Phase 1 physical PDoA characterization

```bash
python analysis/src/workflow.py --phase1     # or --t1-characteristic, --t1-invertibility,
                                             # --t1-elevation, --t1-rotation, --t1-session
```

No learned correction is introduced.  Intervals are 95% grouped-bootstrap
percentiles (`workflow/bootstrap.py`, 1000 replicates, seed 42) resampling
whole configurations (tag, height, distance, rotation), never single samples.
The grouped test subset stays excluded.  `--t1-elevation` reads the existing
`pdoa_stratified_lodo` predictions.

- `t1_1_phase_characteristic/`: fits `pdoa = wrap(scale * eta_expected + offset)`
  on training rows below 15 deg elevation, pooled and per height, and compares
  its inverse with raw and the trigonometric correction on validation.
- `t1_2_invertibility/`: per-tag inverse error below 15 deg elevation and the
  spread of the true phase behind each measured-phase bin.  The useful sector
  is the contiguous range of boresight angles whose inverse MAE is below 10 deg.
- `t1_3_elevation/`: MAE by elevation bin and the matched comparison of
  (h=2, d) against (h=4, 2d), which share the same elevation; h=0 at d and 2d
  controls for the distance change.
- `t1_4_rotation/`: per-rotation circular mean residuals and the share of
  circular variance explained by rotation, below 10 deg elevation.
- `t1_5_session/`: range bias and per-tag PDoA offset per height session at
  d >= 24 m (all heights below 10 deg elevation) and pairwise differences.
  Height and acquisition session are confounded by design.

## Phase 2 generic models, held-out protocols, gating and position

Pre-registered on 2026-10-12, before any phase-2 result was computed:

- **Model `elevation_range`**: target cos/sin of `wrap(pdoa - eta_expected)`;
  design `[1, sin eta, cos eta, sin 2eta, cos 2eta] (x) [1, e, e^2, r_std]`
  (20 columns), with `e = atan(h / rho_hat)` from the measured range and known
  height, and `r_std` the measured range standardized on the training fold.
  Compared with raw, `trigonometric_global` and the two-parameter
  `physical_inverse` (T1.1), all refitted per fold.
- **Success (T2.2)**: at elevation >= 25 deg the MAE falls by >= 30% against
  `trigonometric_global` with a paired grouped-bootstrap interval excluding 0;
  below 25 deg the paired interval must not lie entirely above 0.
- **Gate (T2.3)**: fixed grid over `|pdoa|` in {180, 150, 120, 90, 60, 45, 30}
  deg, measured range in {31, 25, 20, 15, 10} m and estimated elevation in
  {90, 45, 30} deg.  Operating points are the lowest-MAE cells reaching 90, 75
  and 50% coverage of out-of-fold LODO rows.  Position (T2.4) uses the 75% point.

```bash
python analysis/src/workflow.py --phase2
# or individually, in this order:
python analysis/src/workflow.py --loho-distance
python analysis/src/workflow.py --pdoa-elevation lodo --pdoa-elevation loho --pdoa-elevation lobo
python analysis/src/workflow.py --pdoa-gate
python analysis/src/workflow.py --position-elevation lodo --position-elevation loho
```

Outputs: `distance_loho/` (same layout as `distance_lodo/`),
`pdoa_elevation_{lodo,loho,lobo}/` (wide predictions, metrics by elevation
stratum with paired differences against `trigonometric_global`, per-fold
metrics, and the pre-registered verdict in the manifest), `pdoa_gate/`
(complete coverage-error curve and operating points) and
`position_elevation_{lodo,loho}/` (offset range + each phase method, all rows
and gated).  Operating points are selected on the development pool and must be
confirmed on the grouped test (T3.2).

## Phase 3 final grouped-test evaluation (T3.2)

Pre-registered on 2026-10-12, before the grouped test subset was evaluated by
any command.  Models are refitted once on the full development pool (train +
validation) and frozen:

- **Range**: constant horizontal-distance offset on distance-quality-enabled
  development rows.  Reported against raw on test, both with the shared 3 m
  quality mask (primary, as in every earlier benchmark) and on all defined
  horizontal projections (secondary; the mask uses surveyed geometry).
- **PDoA**: primary `physical_inverse` (T1.1 characteristic, scale and offset);
  comparators raw and `trigonometric_global`, refitted on the same pool.
- **Gate**: measured 3D range <= 25 m and elevation from measured range
  <= 45 deg (T2.3 75% operating point); no PDoA limit.
- **Position**: offset ground distance + `physical_inverse` phase through the
  existing geometric reconstruction, all rows and gated; firmware `x_cm`/`y_cm`
  as reference.  Quality-enabled rows only, as in T2.4.

All intervals: 95% grouped bootstrap by configuration.  The command refuses to
run again once `outputs/final_test/manifest.json` reports `complete`;
`--force-final-test` exists only for recovering from a crashed run and is
recorded in the manifest.

```bash
python analysis/src/workflow.py --final-test
```

## Phase 4 static-to-UAV bridge (T4.1, T4.2)

Pre-registered on 2026-10-12, before computing these UAV stratifications.  In
the UAV campaign the node flies and the tags are fixed on the ground.  The
script is diagnostic: it uses RTK ground truth to stratify and, for PDoA, the
oracle mount yaw of `evaluate_uav_pdoa.py`.

```bash
python analysis/src/evaluate_uav_range.py      # adds uwb_time to the predictions
python analysis/src/evaluate_uav_pdoa.py
python analysis/src/evaluate_uav_bridge.py
```

- **Frozen static models**: the 3D range offset (mean `range_error_3d_m` of
  quality-enabled static development rows) and the T3.2 PDoA characteristic.
  Nothing is fitted on UAV data except the oracle mount yaw (PDoA only).
- **T4.1 range**: UAV 3D range residual after the static offset, by elevation
  (<10, 10-25, 25-45, >=45 deg), horizontal distance, speed and
  `uwb_age_s`.  The mechanism is considered shared if, in the static-like
  stratum (speed < 0.5 m/s, elevation < 10 deg, horizontal <= 30 m), the UAV
  MAE is within 2x the static grouped-test MAE of the same 3D offset model.
- **T4.1 PDoA**: residual of the physical inverse against the oracle-mount
  expected phase, by boresight angle, horizontal distance, elevation and
  speed.  Shared mechanism if, in the matched stratum (|boresight| <= 15 deg,
  horizontal <= 25 m, elevation < 45 deg, speed < 0.5 m/s), the UAV MAE is
  within 2x the matched static test MAE (|boresight| <= 15 deg, measured range
  <= 25 m, elevation < 45 deg).
- **T4.2 lag**: RTK re-interpolated at `uwb_time + lag`, lag from -1.0 to
  +1.0 s in 0.05 s steps (narrowed to -0.5..+1.0 s after the timing fix); 3D range residual MAE per lag.  Synchronization is a
  confounder if the best lag reduces MAE by >= 10% against lag 0 with a paired
  interval excluding 0.

Intervals: grouped bootstrap with blocks of flight x tag x 10 s, since UAV
samples are serially correlated.  Outputs in `outputs/uav_bridge/`.

**Known ground-truth timing issue (found by T4.2, 2026-10-12).**  In the drone
logs the RTK latitude/longitude updates at 1 Hz while rows arrive at 10 Hz,
so the logged position is a sample-and-hold staircase.  `interpolate_rtk` in
`evaluate_uav_range.py` interpolates between consecutive rows, which delays
the reference by up to 1 s, and `speed_mps` is 0 whenever two consecutive rows
repeat the same fix (about 88% of the selected records).  Interpolating only
between fix changes, plus a ~0.35-0.40 s reporting latency that transfers
between the two flights, reduces the UAV 3D range MAE from ~1.2 m to ~0.32 m.
UAV results produced before this is fixed in the pipeline are not reliable.

**Ground-truth timing fix (pre-registered 2026-10-12, before rerunning UAV results).**

- RTK x/y are interpolated only between the rows where the logged fix
  changes; `height_fusion_m` likewise between its own change points (no gap
  limit, since a constant fused height is legitimate).  A
  bracket longer than `RTK_MAX_FIX_GAP_S` (1.5 s, i.e. a missed 1 Hz update)
  is not interpolated.  `speed_mps` is the speed of the bracketing fix segment.
- A constant reporting latency is added to every UWB timestamp before
  interpolation.  It is estimated per flight on a 0-1 s grid (25 ms step) by
  minimizing the standard deviation of the raw 3D range residual (invariant to
  a constant range bias), and each flight is evaluated with the latency
  estimated on the *other* flight.  Both estimates are in the manifest.
- The stale per-row `nearest_*` reference columns are removed; downstream UAV
  scripts use the primary (interpolated) geometry.

## UAV bearing and position from PDoA (pre-registered 2026-10-12)

`evaluate_uav_bearing_position.py` derives bearing from the firmware x/y,
which carry the firmware's 93.5 deg phase offset (T0.1); its result is a node
calibration artifact and is kept only as a reference.  The replacement:

```bash
python analysis/src/evaluate_uav_pdoa_position.py
```

- Frozen static models: 3D range offset and PDoA characteristic (T3.2).
- Off-axis angle: `sin(theta) = sign * eta_corrected * r3d_hat / (180 * rho_hat)`
  with `rho_hat = sqrt(r3d_hat^2 - h^2)` from the offset-corrected range and the
  fused height; clipped to [-1, 1]; front hemisphere assumed (|theta| <= 90,
  as in the static test-bed).  The share of truly-rear samples is reported.
- Mount (yaw offset, antenna-order sign): grid search on one flight minimizing
  the corrected-phase residual MAE, evaluated on the other flight; both
  directions reported.  Comparators: raw PDoA (no characteristic), same mount
  protocol.
- Gate: static T2.3 rule, measured range <= 25 m and elevation from measured
  range <= 45 deg.
- Transfer criterion: gated cross-flight position MAE within 2x the static
  grouped-test gated position MAE of the same chain (2.70 m, so <= 5.4 m).
- Intervals: grouped bootstrap over flight x tag x 10 s blocks.

*Deviation from the pre-registration (recorded after the first run).*  The
mount search over both signs is not identifiable: `(offset, sign)` and
`(offset + 180, -sign)` predict identical phase, so the fit picked the
rear-facing twin on one flight (91% of samples then "behind" the node, gated
MAE 19.5 m).  The sign is now fixed to the static antenna-order convention
(`eta = -180 x / r`, node not mounted upside down); the unconstrained fit is
kept in the manifest.  This assumption must be confirmed against the physical
mount.

## UAV reference checks: RTK height and tag survey (pre-registered 2026-10-12)

User facts: node mounted face-up and aligned with the nose (confirms the
static sign convention); tag positions measured with a tape (uncertainty
unknown); RTK telemetry transport latency 10-20 ms.

```bash
python analysis/src/evaluate_uav_survey.py
```

- **R1 height reference.** Replace `height_fusion_m` by
  `rtk_h_m - ground_ellipsoidal_h`, where the ground reference is the median
  of `rtk_h_m - height_fusion_m` on RTK-fixed rows with `height_fusion_m` <
  0.3 m (flight 1 after landing; flight 2 has no fixed ground rows and uses the
  same field reference).  Report UAV 3D range MAE and per-flight bias with
  both references.  Height is a confounder if the inter-flight bias difference
  shrinks by >= 50%.
- **R2 tag survey.** For each tag, estimate its (x, y) from one flight by
  least squares on the 3D range residual after the static offset (node from
  RTK, fixed tag height), and evaluate the other flight with the estimated
  positions; both directions.  Survey error is supported if (a) cross-flight
  range MAE falls by >= 30% against the taped layout with a paired interval
  excluding 0, and (b) the two flights' per-tag estimates agree within half
  of the median correction magnitude.  This is a diagnostic: UAV position
  results keep the taped layout as reference and report sensitivity to it.

- **R3 lever arm and layout translation** (added after R1/R2 failed, before
  running it; motivated by R2's per-tag corrections all pointing towards -y).
  Fit, on one flight, a body-frame lever arm from the RTK reference point to
  the node antenna (forward, right; rotated by yaw) plus a common translation
  of the taped layout (dx, dy), by least squares on the 3D range residual;
  evaluate on the other flight.  Same criterion as R2: >= 30% cross-flight
  MAE reduction with paired interval excluding 0, and the two flights'
  parameter vectors agreeing within half of their median magnitude.

- **R4 RTK degradation before fix loss (post hoc, declared after inspecting
  the data).**  Flight 2 loses the RTK fix at ~640 s (status 50 -> 16, never
  recovered).  In the preceding minutes, while status is still 50, the range
  residual drifts to about -0.6 m and `rtk_h_m - height_fusion_m` departs from
  its flight median.  R4 reports range MAE and per-flight bias excluding fixed
  samples within 60, 120 and 180 s before a fix loss.  Because the window was
  chosen after seeing the drift, R4 is a sensitivity analysis, not a result;
  the primary UAV numbers keep all status-50 samples.

*User facts after R1-R4 (2026-10-12).*  The node sits on the aircraft centre
line, offset towards the nose; the 0.37 m lateral lever arm fitted by R3 is
not physical and R3 is therefore read as absorbing other errors, not as a
mount estimate.  Batteries were hot-swapped between flights (no node reboot).
The RTK solution is limited to 1 Hz by the aircraft firmware.
The forward offset of the node antenna from the aircraft centre is treated as
negligible (user assessment); no lever arm is applied.  The RTK quality
indicator is `rtk_status` itself (50 fixed in flight 1 throughout; in flight 2
50, then 16 after the fix loss); R4 shows that the range residual degrades
before that indicator changes.

## Gate comparison on windows (pre-registered 2026-10-13)

Question: is the softer observable gate (measured range <= 25 m, elevation
from measured range <= 45 deg; T2.3) comparable to the draft's window gate
(circular std of raw PDoA <= 10 deg)?

```bash
python analysis/src/windowed_position.py --window-size 10   # also 20, 30
```

- Same records for every gate: non-overlapping windows of K consecutive
  samples per run/tag on the development pool, out-of-fold LODO quadratic
  range and global-trigonometric PDoA (`geometry_joint_windowK`).
- Gates: none, std <= 10 deg, std <= 20 deg, range+elevation, range+elevation
  and std <= 10 deg.  Window range/elevation use the window mean of measured
  3D range.
- Decision (primary K = 20): adopt the range gate in the paper if its
  position MAE is <= 1.25x the std <= 10 deg MAE; otherwise keep the window
  gate and report the range gate as the per-sample alternative.  Intervals:
  grouped bootstrap by configuration.

*Outcome (2026-10-13).*  K = 20: range gate 2.10 m [1.81, 2.44] at 78.8%
coverage vs std <= 10 deg 0.83 m [0.70, 0.98] at 53.8% (ratio 2.5 > 1.25), so
the window gate is kept.  Not pre-registered as a candidate but computed in the
same run: std <= 20 deg gives 0.85 m [0.75, 0.97] at 69.5%, and range + std <=
10 deg gives 0.67 m at 45.6%.  K = 10 and 30 show the same ordering.
