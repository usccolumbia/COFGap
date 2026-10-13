# Curated CIF screening results

The xTB3 pipeline was run against all 984 CIFs in `curated_cifs`.

## Coverage

- Input CIFs: 984
- Structural descriptors successful: 982
- Complete deCOFpose fragment folders produced: 819
- All three required xTB gaps and final predictions: 511
- Unsupported/missing electronic inputs: 473
- Predicted within the 1.5-2.5 eV PV window: 136

Failures were retained in `predictions.csv` with `MISSING_XTB` status. They
were not assigned predictions using imputed electronic gaps.

## Predicted distribution

- Minimum predicted gap: 1.338 eV
- Median predicted gap: 3.044 eV
- Mean predicted gap: 2.845 eV
- Maximum predicted gap: 3.945 eV

The candidates closest to the center of the PV window (2.0 eV) are:

| CIF | Predicted gap (eV) |
|---|---:|
| 19170N2.cif | 2.000 |
| 20540N2.cif | 1.999 |
| 19420N3.cif | 1.997 |
| 20171N2.cif | 2.006 |
| 20000N2.cif | 1.991 |
| 20280N2.cif | 1.991 |
| 22142N2.cif | 2.010 |
| 21310N2.cif | 2.010 |
| 19573N2.cif | 2.010 |
| 21860N2.cif | 2.011 |

## Qualification

These are model predictions, not measured or DFT-validated gaps. The xTB3
model's family-grouped validation was R2=0.591 and MAE=0.433 eV. Candidate
ordering near 2.0 eV is more defensible than interpreting the third decimal
place. Surface area used a 200-point-per-atom Monte Carlo estimate for this
large screen; it is intentionally lower precision than the training workflow.
Top candidates should have surface area recomputed at 2000+ samples and then
be evaluated with an uncertainty/applicability-domain check before synthesis
or expensive electronic-structure validation.
