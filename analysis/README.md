# Measurement-analysis workflow

This directory keeps the reproducible analysis workflow separate from field
acquisition and from paper-specific plotting code.

The existing `src/main.py` normalizes the static campaigns and exports the
descriptive tables already used by the manuscript.
The files under `src/workflow/` prepare the next stage: controlled benchmark
experiments for range correction, PDoA-phase correction, and derived position
evaluation.

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
```

Generated files are intentionally ignored by Git.
Each benchmark run writes a small JSON manifest and CSV files, so plots and
paper tables can be regenerated from a stable tabular interface rather than
from model-specific objects.

## Planned benchmark sequence

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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --prepare --plan
```

This command is safe: it creates only ignored, header-only output files and a
benchmark plan/manifest.
Model fitting and the additional holdout protocols are deliberately pending.
The exported column contracts are already fixed in `workflow/contracts.py`.

## First step: static-data audit

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --audit-static
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --split-static
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-offset
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-linear
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-quadratic
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-forest
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --compare-distance
/home/fra/pyvenv/bin/python analysis/src/workflow.py --lodo-distance
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
/home/fra/pyvenv/bin/python analysis/src/plot_distance_lodo.py
```

Exports LODO range plots: a compact overall comparison separates simple
calibration candidates from nonlinear baselines, while a second plot reports
MAE at each held-out distance.

## PDoA baseline and circular offset

## Raw PDoA diagnostics by tag, height, and rotation

```bash
/home/fra/pyvenv/bin/python analysis/src/plot_pdoa_diagnostics.py
```

Exports raw, circular PDoA-residual summaries without applying a correction.
`csv/pdoa_raw_by_tag_height.csv` separates h0, h2, and h4 for each tag; its
companion PDF makes the deployment-bearing pattern visible. The rotation CSV
and PDF pool the three heights and instead compare the four known tag rotations.
These diagnostics keep every finite phase record and are deliberately separate
from range-quality filtering.

## PDoA stratified calibration at unseen distances

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --lodo-phase-stratified
```

Evaluates circular-offset and two-harmonic trigonometric corrections with each
of four known deployment conditions: global, tag/bearing, tag rotation, and the
joint tag/bearing--rotation condition. Each of the 15 nominal distances is held
out in turn across every height and rotation. In this fixed test-bed, tag ID and
bearing are equivalent labels; neither ground-truth distance nor position is an
input feature. Outputs are in `outputs/pdoa_stratified_lodo/`.

```bash
/home/fra/pyvenv/bin/python analysis/src/plot_phase_stratified.py
```

Exports overall MAE/RMSE and tag-level MAE PDF plots, with the latter directly
contrasting raw, global-trigonometric, bearing-trigonometric, and joint models.

## Generic position reconstruction at unseen distances

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --evaluate-generic-position
```

Combines matching out-of-fold range and PDoA predictions to compare raw firmware
coordinates with geometric reconstructions. The generic chain uses only range,
PDoA, and known height difference. Tag, bearing, rotation, and ground-truth
coordinates never select a correction. The positive-y half-plane is a known
constraint of the static test-bed; infeasible PDoA projections are explicitly
clipped to the physical disk and their rate is reported.

```bash
/home/fra/pyvenv/bin/python analysis/src/plot_generic_position.py
```

Exports overall and per-tag PDF comparisons of firmware coordinates and the
generic geometric reconstructions.

## Continuous generic PDoA correction at an unseen bearing

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --lobo-phase-continuous
```

Holds out one complete tag/bearing at a time. The continuous models receive
only measured PDoA, measured range, and known height difference; tag, bearing,
and rotation are deliberately excluded. The fixed harmonic model and CUDA MLP
both predict the expected PDoA phase as a continuous quantity.

```bash
/home/fra/pyvenv/bin/python analysis/src/plot_phase_continuous.py
```

Exports overall and per-held-bearing PDF comparisons for the continuous LOBO benchmark.

## Informative sector-classifier position experiment

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --evaluate-sector-position
```

Learns a nine-sector classifier from observed UWB features and selects a bearing-specific PDoA expert without receiving the true sector at inference. It is evaluated LODO and exported separately because it is a fixed-sector diagnostic, not a continuous generic model.

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-phase-offset
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-phase-trig
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --compare-phase
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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-phase-svr
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
/home/fra/pyvenv/bin/python -m pip install -r analysis/requirements-gpu.txt
```

This prepares PyTorch for future GPU models; existing scikit-learn SVR/MLP
implementations still execute on CPU. Installing PyTorch does not switch their
backend. GPU experiments will require a separate implementation and the same
training/validation protocol.

Verify CUDA forward/backward operations and optimizer updates with:

```bash
/home/fra/pyvenv/bin/python analysis/src/check_gpu.py
```

The synthetic check writes versions, device information and loss values to
`outputs/gpu_check.json`; it is not a dataset experiment or a paper result.

Once that check passes, run the GPU phase comparison with:

```bash
/home/fra/pyvenv/bin/python analysis/src/workflow.py --fit-phase-torch
```

It trains two fixed PyTorch MLPs on CUDA: phase-only and enriched. Both predict
the cosine/sine PDoA residual using training-only standardization, two hidden
layers (64, 32), AdamW, 200 epochs, batch size 4096, and seed 42. Validation is
never used for early stopping or parameter selection. Artifacts, metrics and
provenance are written to `outputs/pdoa_torch_gpu/`.
