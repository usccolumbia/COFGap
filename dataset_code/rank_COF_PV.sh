python cof_cif2smiles.py CoRE-COFs_DT591-v5.0/ --out cof_smiles.csv --cap star
python pv_cof_screen.py cof_smiles.csv --out pv_ranked.csv          # instant
python pv_cof_screen.py cof_smiles.csv --out pv_ranked_xtb.csv --xtb    # +seconds/COF


# regenerate SMILES with H-capping instead of [*]
python cof_cif2smiles.py CoRE-COFs_DT591-v5.0// --out cof_smiles_H.csv --cap H

# then screen that new CSV (no --cap here)
python pv_cof_screen.py cof_smiles_H.csv --out pv_ranked_H.csv