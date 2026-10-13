# COFGap: DFT-free band-gap prediction of covalent organic frameworks

Code, derived data, and results for:

> J. Hu, P. Sarker, J. Hu, T. Wei. *DFT-free band gap prediction of COFs using foundation models and topological and fragment features.* (submitted)

COFGap predicts COF band gaps from GFN2-xTB electronic descriptors of isolated
node, linker, and node+linker fragments plus framework structural/topological
descriptors, using the TabPFN tabular foundation model. Headline result:
TabPFN v3.5, 216 CoRE-COF v5.0 structures, 72 families, five-fold
family-grouped CV: pooled out-of-fold R² = 0.656, MAE = 0.373 eV.

## Layout

| Folder | Content (manuscript item) |
|---|---|
| `dataset_code/` | CoRE-COF band-gap table, fragmentation (`cof_cif2smiles.py`) and feature scripts |
| `DFT_cof_bandgap/` | Internal DFT/ORCA fragment baseline and fragment geometries (`fragments/`, xyz + ORCA in/out) |
| `DFT_replaced_xTB1` | Structure-only and single-xTB-gap baselines; `04_rerun_216/` = Table 3 xTB1 rows on the 216-COF cohort |
| `DFT_replaced_xTB2` | Rich GFN2-xTB extraction (`04_rich_xTB_extraction/worker.py`, tblite) and models |
| `DFT_replaced_xTB3` | Compact 17-feature model, PBE0 reference (Ernst et al. Zenodo 14855145), calibration tests |
| `DFT_replaced_xTB4` | Physics-informed 92-feature table (`01_feature_table/xtb4_features.csv`), family labels, model comparison (Table 5) |
| `DFT_replaced_xTB5_TabPFN_V3` | TabPFN v3 benchmark |
| `DFT_replaced_xTB7_validation_protocols` | Random 80/10/10 holdout split assignments |
| `DFT_replaced_xTB6/8/9_Wei2025*` | External 725-COF Wei-2025 cohort (manifests, xTB features, validation) |
| `DFT_replaced_xTB10_TabPFN_V3_5` | **Primary results**: TabPFN v3.5 grouped CV (`run.py`), random holdout, Wei-725 check |
| `figure_scripts/`, `figures/` | Scripts that regenerate Figures 3, 4 and 7 |

Every grouped-CV run uses `sklearn.model_selection.GroupKFold(5)` on the
`family` column of `xtb4_features.csv`, so fold assignments are deterministic;
`oof_predictions.csv` files record each COF's fold and prediction.

## Reproducing the main result

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
TABPFN_DISABLE_TELEMETRY=1 python DFT_replaced_xTB10_TabPFN_V3_5/run.py
python figure_scripts/make_fig4_parity_v3_5.py
```

TabPFN v3/v3.5 weights are not redistributed here. They are downloaded by the
`tabpfn` package after you accept the Prior Labs licence (https://ux.priorlabs.ai)
and set `TABPFN_TOKEN`.

## Structures (not included)

CIF files are not redistributed. Obtain them from the original sources:

- CoRE-COF v5.0: https://github.com/core-cof/CoRE-COF-Database
- CURATED-COFs (for the Wei-2025 cohort): https://archive.materialscloud.org/
- Wei et al., *J. Phys. Chem. C* 2025, 129, 7126 (DOI 10.1021/acs.jpcc.5c01307), Table S1

The manifests (`*_manifest.csv`) list the exact file names used.

## Third-party data

- PBE0 fragment gaps in `DFT_replaced_xTB3/00_Zenodo_14855145/` are from
  M. Ernst et al., Zenodo record 14855145 (CC BY 4.0).
- Reference band gaps are from CoRE-COF v5.0.

## Licence

Code: MIT (see `LICENSE`). Derived data produced in this work: CC BY 4.0.
Please cite the paper and this archive (see `CITATION.cff`).
