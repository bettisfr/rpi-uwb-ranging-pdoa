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
/home/fra/pyvenv/bin/python analysis/src/workflow.py --prepare --plan static
```

This command is safe: it creates only ignored, header-only output files and a
benchmark plan/manifest.
Implementation of the data adapter and regressors is deliberately pending.
The exported column contracts are already fixed in `workflow/contracts.py`.
