# xTB5: TabPFN V3 result

## Controlled comparison

This is a direct model-version comparison. Both versions used the xTB4
physics-informed feature table, the same 216 complete-xTB COFs, the same 72
family labels, the same five `GroupKFold` splits, median imputation fitted
within each training fold, and eight TabPFN estimators per fold.

| Model version | R2 | MAE (eV) |
|---|---:|---:|
| TabPFN V2 | 0.6235 | 0.3814 |
| TabPFN V3 | **0.6446** | **0.3808** |

## Change from V2 to V3

- R2 improves by **0.0211** (0.6235 → 0.6446).
- MAE improves by **0.0005 eV** (0.3814 → 0.3808 eV), which is practically
  unchanged at this precision.

V3 therefore captures more of the variation across the held-out COFs, but the
average absolute band-gap error does not materially change. The result remains
a 216-COF complete-xTB-cohort score; it is not yet a 315-COF deployment score.

All folds passed the no-row-overlap and no-family-overlap checks. See
`01_benchmark_v3/metrics.json` and `01_benchmark_v3/oof_predictions.csv`.
