# DFT_replaced_xTB3 results

## Data assembled

- 945 released PBE0 fragment-label occurrences.
- 732 successful xTB/PBE0 pairs covering 355 unique canonical fragments.
- Pair coverage: 257 nodes, 258 linkers, and 217 node+linker fragments.
- Final fair COF comparison: 216 COFs from 72 held-out families.

## Fragment-level prediction

| Method | Grouped R2 | MAE (eV) |
|---|---:|---:|
| Raw xTB gap used as PBE0 gap | -3.709 | 2.262 |
| Linear xTB calibration | 0.453 | 0.644 |
| Ridge delta calibration | 0.449 | 0.649 |
| ExtraTrees delta calibration | 0.326 | 0.698 |
| Direct SMILES/RDKit surrogate | **0.771** | **0.398** |

Fragment folds are grouped by canonical SMILES, so repeated fragments cannot
occur in both calibration training and test sets.

## Final COF band-gap prediction

All variants below use the same 216 COFs, structure descriptors, and
family-grouped folds.

| Electronic feature source | Best grouped R2 | MAE (eV) |
|---|---:|---:|
| Raw node/linker/nodelinker xTB gaps | **0.591** | 0.433 |
| Delta-calibrated xTB gaps | 0.362 | 0.546 |
| Direct fragment surrogate | 0.516 | 0.446 |
| Original DFT fragment gaps | 0.625 | **0.410** |

## Interpretation

The direct surrogate is the best way to reproduce PBE0 fragment gaps without
DFT, but exact reproduction of fragment PBE0 gaps is not identical to optimal
framework band-gap prediction. Raw xTB gaps retain useful nonlinear ranking
information that the final random forest can combine with stacking and
structure. Explicit calibration compresses that signal and propagates
out-of-fragment calibration error into the second model, so it performs worse.

xTB3 improves the best DFT-free grouped R2 from the xTB2 rich-feature result
of 0.575 to 0.591. It remains below the same-population DFT reference at 0.625.
The performance gap is now 0.034 R2 and 0.023 eV MAE.

The packaged model is `04_final_COF_model/DFT_replaced_xTB3_model.joblib`. It
uses no DFT-derived inputs, but requires raw GFN2-xTB gaps for the node,
linker, and node+linker fragments plus inexpensive structural descriptors.
