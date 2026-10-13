# Experiment results

All scores are out-of-fold predictions from five-fold family-grouped CV.

## Best current result

- Experiment: `02_xtb_electronic/results_complete_cases`
- Model: `extra_trees`
- Samples: 271 (80 families)
- R2: 0.504
- MAE: 0.516 eV

## Common structural baseline

- Model: `extra_trees`
- R2: 0.358
- MAE: 0.584 eV

See `comparison.csv` for every model/result set. Complete-case xTB results use
fewer samples than all-row results and should not be compared without noting
that population change. The all-row xTB experiment is the fair deployable
comparison against the 315-row structural baseline.
