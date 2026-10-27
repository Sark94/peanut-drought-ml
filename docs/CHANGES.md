# Source-to-package mapping

Source: October 9 notebook confirmed unchanged by the author. Cell numbers below
are zero-based notebook indices (visible notebook headings sometimes differ).

| Source | Package | Treatment |
|---|---|---|
| 0–4 | data.py / make_cv | Explicit file arguments; metadata excluded; labels explicit; original CV defaults retained |
| 5, 14 | models.py | All nine pipelines and complete grids retained; fresh clones returned |
| 6–7 | evaluation.py | Duplicate function collapsed; output directory/jobs configurable; denominator uses actual fold count rather than fixed 50 |
| 13 | selectors.py | Original PCA loading-weighted gene-ranking algorithm retained |
| 19–22 | consensus.py / example | Strict and exploratory recurrence rules exposed; deterministic tie ordering by gene name |
| 23–24 | hyperparameter_frequency | Correct combination-based grouping retained; JSON parsing replaces eval fallback |
| 25 | fit_final(mode='tuned') | Tunes classifier only on explicitly supplied ordered panel |
| 26 | fit_final(mode='fixed') | Original eight-gene panel and C=10, L2/liblinear specification retained |
| 29 | removed | Debug cell referenced undefined frozen_genes |
| 31–32 | ModelBundle / validate_external | Model choice explicit; training gene order and label mapping travel with model |
| 34–36 | plotting.py | Counts calculated from predictions instead of hard-coded matrices |
| 37 | bootstrap_ci | Same unstratified percentile approach; optional shared RNG reproduces sequential cohort usage |

The two final-model routes give identical predictions on the supplied inputs:
the tuned model selects C=10, class_weight=None, matching the fixed specification.
The exported notebook's confusion counts are reproduced for both external cohorts.
This does not establish portability to different data processing or different populations.

Additional behavior: reject invalid input, abort failed grid candidates rather than
silently ignore errors, preserve the original notebook for audit, log environment,
and support a pip-installable package and CLI. No analysis runs at import time.

No new scientific algorithms, external data, or automatic raw-count normalization
were introduced. Previously computed values in the source notebook remain archival;
they are not substituted for computed results in package functions.
