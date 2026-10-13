# xTB7: Effect of validation protocol (TabPFN V3)

All xTB5/xTB7 rows use the same 216 complete-xTB COFs and 92 usable DFT-free
xTB4 feature columns. Random split results use seed 17.

| Evaluation | Split / model | Held-out prediction population | R2 | MAE (eV) | Family overlap? |
|---|---|---:|---:|---:|---|
| DFT reference benchmark | RF + original DFT fragment gaps | 216, grouped 5-fold | 0.625 | 0.410 | No |
| xTB4 | TabPFN V2, grouped 5-fold | 216 | 0.624 | 0.381 | No |
| xTB5 | TabPFN V3, grouped 5-fold | 216 | **0.645** | 0.381 | **No** |
| xTB7 random split | TabPFN V3, validation partition | 22 | 0.726 | 0.397 | Yes (13 shared family labels with train) |
| xTB7 random split | TabPFN V3, test partition | 22 | **0.747** | **0.300** | Yes (13 shared family labels with train) |

## Interpretation

The random 80/10/10 test score is optimistic for novel-family prediction:
members of 13 family labels appear in both training and test partitions. The
random test set also contains only 22 COFs, so its metrics have high sampling
variability. Family-grouped cross-validation is the primary reported result
because it excludes every defined family from its corresponding training fold,
tests all 216 COFs once, and has zero row/family overlap.

The Wei-2025 external run is reported separately in `DFT_replaced_xTB6_Wei2025_external`.
Its labels use a different target convention, so it is an external
target-domain-shift diagnostic rather than a directly comparable accuracy row.
