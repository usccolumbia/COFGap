# DFT_replaced_xTB3

Version 3 starts from the primary Zenodo release for Ernst et al., DOI
`10.5281/zenodo.14855145`, and builds fragment-level PBE0 calibration data.

The raw release is stored in `00_Zenodo_14855145`. Run the audit with:

```bash
../../.venv312/bin/python 00_Zenodo_14855145/audit_zenodo.py
```

The audit converts the 315-row COF table into 945 node/linker/nodelinker label
occurrences, preserves separate PBE0 HOMO and LUMO values, canonicalizes the
SMILES, identifies repeated fragments, and checks for matching local
deCOFpose fragment XYZ geometries.

The Zenodo record contains the full labels but only example calculation files
and geometries for COF 1. The matching all-COF fragment geometries already
exist locally under `DFT_cof_bandgap/fragments`; fragmentation itself does not
need to be changed or rerun with DFT.

## Pipeline

```bash
../../.venv312/bin/python 01_calibration_dataset/build.py
../../.venv312/bin/python 02_delta_calibration/run.py
../../.venv312/bin/python 03_direct_surrogate/run.py
../../.venv312/bin/python 04_final_COF_model/run.py
```

- `01_calibration_dataset` pairs successful xTB calculations with Zenodo PBE0 labels.
- `02_delta_calibration` learns `PBE0 gap - xTB gap` using fragment-grouped CV.
- `03_direct_surrogate` predicts PBE0 gaps directly from SMILES/RDKit features.
- `04_final_COF_model` compares raw xTB, corrected xTB, direct surrogate, and
  original DFT fragment features on the same COFs and family-grouped folds.
