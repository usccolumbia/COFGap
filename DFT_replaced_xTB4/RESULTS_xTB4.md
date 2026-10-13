# DFT_replaced_xTB4 results

All results use the same 216 COFs, 72 families, target values, and five
family-grouped folds used for xTB3.
Grouped cross validation.

| Feature set | Model | R2 | MAE (eV) |
|---|---|---:|---:|
| xTB3 baseline | Random forest | 0.591 | 0.432 |
| xTB3 baseline | TabPFN v2 | 0.578 | 0.450 |
| Physics-informed xTB4 | Random forest | 0.591 | 0.432 |
| Physics-informed xTB4 | ExtraTrees | 0.596 | 0.405 |
| Physics-informed xTB4 | TabPFN v2 | **0.624** | **0.381** |
| Physics-informed xTB4 | 75% TabPFN + 25% ExtraTrees | 0.635 | **0.377** |
| Physics-informed xTB4 | 75% TabPFN + 25% RF | **0.640** | 0.380 |

The standalone TabPFN result is the primary algorithm comparison. The fixed
ensembles are useful deployed candidates, but because their weights were
examined after viewing these cross-validation results, they should be
confirmed on a locked external set or with nested grouped CV before claiming
0.640 as the definitive generalization score.

The new descriptors include separate frontier energies, vertical IP/EA,
electronegativity/hardness, dipole, charge polarization, bond-order summaries,
fragment-size ratios, node-linker orbital offsets, gap lowering on bonding,
composition fractions, pore/atom normalization, and lattice anisotropy.

TabPFN 8.3.0 was pinned to official `ModelVersion.V2`. The newer default 2.6
weights were not used because they require the user to personally accept a
non-commercial model-weight license and supply `TABPFN_TOKEN`.
