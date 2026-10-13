# xTB8: Wei-2025 725-COF xTB–TabPFN V3 benchmark

## Coverage

| Wei-2025 labeled COFs | Complete rich-xTB feature rows | Coverage |
|---:|---:|---:|
| 725 | 476 | 65.7% |

The 249 excluded COFs lack at least one required node/linker/node+linker xTB
record. Of the 476 usable rows, 156 reuse rich-xTB results for identical CoRE
v5.0 positional structures already present in xTB5; 320 are new Wei records
with successful fresh rich-xTB calculations.

## Validation results

| Dataset / protocol | TabPFN model | Train / validation / test | Evaluation COFs | R2 | MAE (eV) | Spearman rho | Family overlap? |
|---|---|---:|---:|---:|---:|---:|---|
| xTB5 original cohort, 5-fold grouped | V3 | fold-specific | 216 | 0.645 | 0.381 | n/a | No |
| Wei xTB8 random split (seed 17), validation | V3 | 380 / 48 / 48 | 48 | 0.285 | 0.274 | 0.707 | Yes |
| Wei xTB8 random split (seed 17), test | V3 | 380 / 48 / 48 | 48 | **0.706** | **0.185** | **0.828** | Yes; 8 shared family proxies |
| Wei xTB8 family-proxy grouped 5-fold | V3 | 380–381 / n.a. / 95–96 | 476 | **0.403** | **0.286** | **0.674** | No |

## Interpretation

The Wei random hold-out split is a single 48-COF test set and shares eight
family-proxy labels with training. Its R2 is therefore an optimistic and
high-variance interpolation result; the validation and test subsets already
show substantial split-to-split variation (0.285 vs 0.706 R2).

The grouped value is the appropriate primary Wei score because it tests every
complete case once while preventing the available chemistry-family proxy from
crossing between training and test folds. The Wei release does not include the
linker-class family labels used by xTB5, so xTB8's proxy is source collection +
capped B/N/O composition. It is a stricter-than-random split, but it is not
identical to the original xTB5 family definition.

The Wei numbers should not be compared directly to the original 216-COF xTB5
score as evidence of a model regression or improvement: they use a different
label source/convention, a different COF population, a different coverage
subset, and a family proxy rather than the original linker-aware family label.
