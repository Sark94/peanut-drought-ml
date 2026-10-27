# Validation — October 27, 2026

## Completed

- 15 automated tests passed (Python 3.12; installed runtime scikit-learn 1.8.0,
  NumPy 2.3.5, pandas 2.2.3). Deprecation and metric warnings were emitted;
  no tests failed. The package currently caps scikit-learn below 1.9.
- All nine original model/selector combinations exercised with small nested-CV
  grids. These tests verify operation, not original full-grid scientific results.
- Original notebook nested-CV function and packaged function compared on the
  same synthetic data with 50 outer folds and a reduced LR grid: all six returned
  tables matched exactly. PCA selector compared directly with the source class.
- Tests cover explicit class encoding, malformed data rejection, gene-name
  alignment, model save/load, no refitting during validation, consensus selection,
  bootstrap reproducibility, and final-model tuning.
- Synthetic demonstration ran successfully.
- CLI final-model training and external validation ran successfully; the CLI
  smoke test used 50 bootstrap replicates, not the default 5,000.
- On supplied real data, using notebook versions NumPy 2.0.2, pandas 2.2.2 and
  scikit-learn 1.6.1: 42 × 276 discovery inputs loaded; final LR tuning selected
  C=10 and class_weight=None. Fixed and tuned predictions matched the direct
  source-model implementation to relative tolerance 1e-12.
- Jiang external confusion counts: TN=6, FP=0, FN=1, TP=17.
  Sun external confusion counts: TN=6, FP=0, FN=2, TP=10.
  Both reproduce the recorded notebook counts. See `real_data_checks/`.

## Not claimed

The entire nine-model, full-grid nested cross-validation experiment was not rerun.
The full 5,000-resample confidence interval analysis was not rerun on both real
cohorts. The package tests establish implementation consistency for the checks
listed above, not independent biological validation or absence of upstream leakage.
Source results and the original notebook are preserved for comparison.

## Run checks yourself

From the package root after installing the dev extra:

```bash
python -m pytest -q
python examples/demo.py
python docs/verify_real.py
```

The last command uses the three CSVs under `provenance/data/`. Use the pinned
notebook environment for the closest numerical reproduction. Figures in
`real_data_checks/` are computed from model predictions, not hard-coded counts.
