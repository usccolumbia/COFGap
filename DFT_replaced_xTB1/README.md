# DFT_replaced_xTB1

Version 1 is the preserved baseline implementation using the original single
GFN2-xTB fragment-gap column.

Experiments:

- `00_structure_baseline`
- `01_rdkit_molecular`
- `02_xtb_electronic`
- `03_combined_residual`

The best equal-population result was the all-row xTB model with grouped
R2=0.406 and MAE=0.552 eV on 315 COFs. The complete-case result was R2=0.504
and MAE=0.516 eV on 271 COFs.

Run with:

```bash
../../.venv312/bin/python run_all.py
```
