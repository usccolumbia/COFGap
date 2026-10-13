# DFT_replaced_xTB2 results

All comparisons below use five-fold family-grouped cross-validation. The fair
rich-xTB versus DFT comparison uses the same 216 complete-case COFs and 72
families.

| Model | DFT inputs used for prediction | R2 | MAE (eV) |
|---|---:|---:|---:|
| Rich xTB only | No | 0.357 | 0.549 |
| Rich xTB + structure | No | 0.573 | 0.438 |
| Rich xTB + structure + RDKit | No | 0.575 | 0.432 |
| DFT fragment gaps only | Yes | 0.281 | 0.610 |
| DFT fragment gaps + structure | Yes | 0.596 | 0.413 |

The xTB2 model is close but has not yet beaten the DFT reference on both
metrics. Its R2 is lower by 0.021 and its MAE is higher by 0.019 eV. A
physics-curated xTB subset reached exploratory R2 around 0.600, but that
feature/hyperparameter choice was inspected on the evaluation folds and is
therefore not promoted as a final score. It must be confirmed with nested
grouped model selection or a locked external test set.

R2 measures explained variation relative to predicting the mean. Higher is
better; 1.0 is perfect and 0.0 means no improvement over the mean. MAE is the
average absolute error in electron-volts. Lower is better and is usually more
directly interpretable for screening decisions.

The rich extraction generated 39 properties for each of node, linker, and
nodelinker, plus nine explicit node-linker interaction features. All three
fragment calculations succeeded for 216 of the 315 labeled COFs.
