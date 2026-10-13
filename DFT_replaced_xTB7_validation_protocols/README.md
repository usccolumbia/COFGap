# xTB7: Validation-protocol comparison

This experiment evaluates the unchanged xTB5 TabPFN V3 representation on the
same 216 complete-xTB COFs under two different evaluation protocols:

1. a single random 80/10/10 train/validation/test split (random state 17); and
2. five-fold family-grouped cross-validation.

The random split reports validation and untouched-test metrics separately.
The grouped result reports every COF's one out-of-fold prediction. The two
protocols answer different questions: random splitting estimates interpolation
among similar COFs, while family grouping is stricter for chemical-family
generalization.
