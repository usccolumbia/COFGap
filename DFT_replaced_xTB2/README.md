# DFT_replaced_xTB2

Version 2 expands the single xTB gap into separate node, linker, and
nodelinker electronic properties. Version 1 remains unchanged in the adjacent
`DFT_replaced_xTB1` folder.

These experiments compare progressively richer, inexpensive alternatives to
fragment-DFT features.  Every experiment has its own folder and result files.

Run all experiments from this directory:

```bash
../../.venv312/bin/python run_all.py
```

The rich extraction/model stages are run separately because extraction is
expensive and resumable:

```bash
../../.venv312/bin/python 04_rich_xTB_extraction/extract.py --labeled-only --retry-errors --jobs 8
../../.venv312/bin/python 05_rich_xTB_model/run.py
../../.venv312/bin/python 06_DFT_reference_benchmark/run.py
```

Experiments:

0. `00_structure_baseline`: the common inexpensive structural feature baseline.
1. `01_rdkit_molecular`: framework geometry plus RDKit molecular descriptors
   and separate fingerprints for the two largest molecular fragments.
2. `02_xtb_electronic`: framework geometry plus the available GFN2-xTB
   fragment gap.  It reports both all-row (with a missing-value indicator) and
   complete-case performance.
3. `03_combined_residual`: combines geometry, RDKit, fingerprints, and xTB. It
   also learns the correction `DFT gap - xTB gap` on complete xTB cases.
4. `04_rich_xTB_extraction`: computes a comprehensive GFN2-xTB feature table
   from the real CIF geometry, including separate frontier orbitals, vertical
   IP/EA, charges, dipole, quadrupole, and bond-order summaries.
5. `05_rich_xTB_model`: compares rich-xTB-only, rich-xTB plus structure, and
   rich-xTB plus structure/RDKit models using the same grouped folds.
6. `06_DFT_reference_benchmark`: evaluation-only reference using the original
   DFT fragment features on exactly the rich-xTB complete-case population.
   These features are never included in the deployable xTB models.

All headline scores use the same five family-grouped folds.  Rows from the
same coarse linkage/composition family cannot appear in both training and
test partitions.

The two largest dot-separated fragments are used as reproducible fragment A
and fragment B representations.  They are size-ranked, not chemically
assigned as node and linker; a future deCOFpose assignment can replace this
without changing the evaluation code.

True molecular polarizability is not returned by the `tblite` Python backend.
The rich table therefore includes Delta-SCF IP/EA, chemical hardness,
softness, electronegativity, and electrophilicity, but does not mislabel a
proxy as polarizability.
