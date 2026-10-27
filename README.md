# Peanut Drought ML — 0.1.1

An installable Python package built from Frederick Sarkodie Mensah's
`Peanut_Drought_ML_Reproducible_Final (1).ipynb` (confirmed unchanged October 27, 2026).
Import name: `peanutml`. Distribution name: `peanut-drought-ml`.
This is a tested initial package release, not a PyPI publication or a new scientific method.

## Software authors

1. Frederick Sarkodie Mensah
2. Muhammed Opoku Gyamfi

## Google Colab

Open `examples/PeanutML_Colab_Quickstart.ipynb` in Colab and run its cells in order.
Upload this complete package ZIP when prompted. The notebook installs the package,
runs the synthetic demo and downloads its outputs. Use a fresh Python runtime.

## Install

Use Python 3.10–3.12 in a virtual environment. From this directory:

```bash
python -m pip install .
```


For the closest match to the source notebook, use Python 3.12 and install
`requirements-notebook.txt` before installing this package. The current environment
validation is documented in `docs/VALIDATION.md`; environment versions affect results.
No online services, Colab upload widgets, or API keys are required at runtime.

## Quick example

```bash
python examples/demo.py
```

The demo generates synthetic expression data, performs a reduced LR search, and
writes tables into `demo_results/`. Its output is NOT a research result.

## Your discovery analysis

Input must already be an appropriately transformed numeric expression matrix
(samples as rows, genes as columns), not raw RNA-seq counts. The notebook uses
ComBat-seq/VST inputs prepared outside Python.

Default metadata columns: `Unnamed: 0` (sample ID), `Condition` (Control/Drought),
`Batch` (optional; excluded from predictors). Other metadata must be removed or
excluded by selecting explicit gene columns through the Python API. Labels are
explicitly encoded Control=0, Drought=1. External CSVs default to `SampleID`.
Missing values, infinities, duplicate IDs/genes and unrecognized labels are rejected.

```bash
peanutml cv sark_276_ComBatSeq_VST_for_ML_with_condition.csv --model anova_lr --output results/anova_lr --jobs 4
```

Defaults preserve the notebook: seed 42, outer 5 folds × 10 repeats, inner
4 folds × 3 repeats, balanced-accuracy optimization, and the original parameter grids.
Use `peanutml cv --help` for configurable folds and feature counts. Full grids,
especially Random Forest, are expensive. `--jobs` defaults to 1 to avoid exhausting
shared machines; changing it does not change the search grid.

Nine model keys:
`anova_lr`, `anova_linear_svm`, `anova_rbf_svm`, `anova_rf`, `anova_knn`,
`pca_lr`, `pca_linear_svm`, `pca_rbf_svm`, `pca_knn`.
The Random Forest model is included because it is in the confirmed source notebook.
PCA selection ranks original genes by variance-weighted absolute loadings; it does
not replace expression columns with principal component scores.

Each CV run writes fold metrics, held-out predictions, selected genes, best
hyperparameters, mean/SD summaries, selection frequencies and run configuration.
Selection and scaling remain inside the fitted pipeline at each training split.

## Python API

```python
from peanutml import load_dataset, get_model, make_cv, run_nested_cv

data = load_dataset('sark_276_ComBatSeq_VST_for_ML_with_condition.csv')
pipeline, grid = get_model('anova_lr')
outer, inner = make_cv()
result = run_nested_cv('anova_lr', pipeline, grid, data.X, data.y,
                       data.sample_ids, outer, inner, output_dir='results', n_jobs=4)
```

Run the three consensus pipelines and pass their `gene_frequency` tables to
`consensus_genes` (see `examples/consensus_workflow.py`). The strict notebook rule
is ≥50% recurrence in all three: ANOVA LR, ANOVA linear SVM and PCA linear SVM.
The alternative exploratory rule uses at least two of three. A panel derived from
all discovery samples must not be treated as independently validated by those same samples.

## Fit a final panel and evaluate an external cohort

**Choose the final model explicitly.** The notebook saved a fixed LR with C=10
(cell 26), but its external validation used the tuned estimator from cell 25
(cell 31 assigns `final_lr_search.best_estimator_`). The package preserves both routes:

```bash
peanutml train sark_276_ComBatSeq_VST_for_ML_with_condition.csv --mode fixed --output results/fixed
peanutml train sark_276_ComBatSeq_VST_for_ML_with_condition.csv --mode tuned --genes strict_panel.csv --output results/tuned
peanutml validate Jiang_external_frozen8_VST.csv --model-file results/fixed/model.joblib --output results/jiang
```

`strict_panel.csv` must contain an ordered `Gene` column. Without `--genes`, training
uses the notebook's exact eight-gene fixed panel. To reproduce the tuned route,
supply the exact ordered consensus panel from your discovery run. The package does
not silently select between the two routes. External data are never used to fit,
tune or rescale the model. Validation exports metrics, predictions, bootstrap CIs,
and a confusion-matrix figure calculated from the predictions.

`ModelBundle` stores gene order and class mapping. Extra external columns are ignored;
missing required genes raise an error. The discovery scaler is reused for prediction.
Joblib files are pickle-based: load only your own or trusted models, using compatible
library versions. No trained model is included as the default package model.

## Reproducibility and scientific scope

- Original source notebook and analysis-only code are under `provenance/`.
- `docs/CHANGES.md` maps implementation changes to notebook behavior.
- `docs/VALIDATION.md` distinguishes tested equivalence from unperformed full reruns.
- Fold SD is descriptive, not a confidence interval: repeated folds are dependent.
- The package cannot undo upstream leakage. If the 276-gene candidate list was
  selected using the same outcomes, nested CV is conditional on that preselected
  list and does not independently evaluate the entire discovery workflow.
- Preprocessing, batch correction and any feature filtering must be assessed for
  train/test separation. Random stratified folds do not measure generalization to
  unseen studies; the notebook's `Batch` column is excluded, not used for group splits.
- Bootstrap CIs resample external subjects and condition on the frozen model.
  Small external cohorts and correlated samples limit interpretation. For related
  observations, subject/study-aware splits and resampling require a separate design.
- This package operates on processed expression features; it does not implement
  FASTQ processing, DESeq2, ComBat-seq, VST, or WGCNA.

## Development

```bash
python -m pip install '.[dev]'
python -m pytest -q
python -m build
```

Before public release, confirm the package name, dataset redistribution permissions and final citation metadata.
No public repository or Zenodo record has been changed by creating this package.

The three supplied research CSVs are included in `provenance/data/` for local
reproduction. For example, replace a bare CSV filename above with
`provenance/data/sark_276_ComBatSeq_VST_for_ML_with_condition.csv`.

## License

Package code is available under the MIT license; see LICENSE. Original analysis
material in provenance/ retains its applicable CC BY 4.0 terms. See
LICENSE-NOTICE.txt for scope, source attribution and dataset exceptions.

## How to cite

If this package contributes to your research, please cite the software and the
original analysis that it builds on. This is a scholarly request, not an extra
condition of the MIT license.

**Package citation (local draft; no package DOI assigned yet):**

Mensah, Frederick Sarkodie, and Gyamfi, Muhammed Opoku. Peanut Drought ML
(Version 0.1.1) [Computer software].

After public archiving, cite the package's own version DOI here. The original
analysis DOI below identifies the earlier deposit, not this package release.
CITATION.cff provides the package authors and version in machine-readable form.

**Original analysis:**

Mensah, F. S. (2026). Reproducible Analysis Code for "Integrative Transcriptome
Meta-Analysis, Co-expression Network Analysis, and Machine Learning
Computationally Prioritize Drought-Responsive Candidate Genes in Peanut"
(Version v1) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.23291732
