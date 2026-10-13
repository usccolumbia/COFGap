# DFT-free COF band-gap feature replacement

Goal: keep the DFT band gap as the supervised target, but remove expensive or chemically opaque DFT-derived input descriptors from the model features.

## Removed input features

These columns from `bandgap_CoRE_COF_v5.0_PCAdata.csv` are no longer used as model inputs:

- `Node_HL_gap(eV)` -> old model key `node_hl`
- `Linker_HL_gap(eV)` -> old model key `linker_hl`
- `Node+Linker_HL_gap(eV)` -> old model key `nl_hl`

Reason: these are fragment electronic descriptors that require quantum chemistry-style calculations and can leak a close proxy for the target band gap.

## Replacement feature groups

### A_atomic

Cheap composition and heteroatom descriptors:

- `n_count`
- `b_count`
- `o_count`
- `hetero_count`
- `n_frac`
- `b_frac`
- `o_frac`
- `hetero_frac`

### S_structure

Geometric and topological descriptors from CIF/screener output:

- `density`
- `sacc`
- `volume`
- `n_atoms`
- `volume_per_atom`
- `atom_density`
- `layered`
- `stack_len`
- `n_pbc_bonds_cut`
- `pbc_bond_density`
- `n_unique_fragments`

### C_chemistry

Interpretable chemistry flags from `pv_ranked_H.csv`:

- `conj`
- `sp3`
- `link_score`
- `has_donor`
- `has_acceptor`
- `is_DA`
- `has_metal`

## Updated model feature sets

- `A_atomic`
- `S_structure`
- `C_chemistry`
- `A+S`
- `A+S+C`

The grouped-CV model also includes:

- `minimal(stack+density)`

## Generated DFT-free table

Run:

```bash
./.venv312/bin/python make_dftfree_feature_table.py
```

Output:

- `bandgap_CoRE_COF_v5.0_DFTfree_features.csv`

This table keeps `band_gap_eV` as the target but excludes the three fragment HL-gap input columns.
