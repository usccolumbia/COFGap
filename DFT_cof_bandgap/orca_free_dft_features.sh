# 0. one-off
# pip install pyzeo          # tblite you already have
'''
# 1. surface area  (~75 s for all 591)
python make_dft_features.py sacc --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out sacc.csv

# 2. cut into node / linker / node+linker
# python make_dft_features.py fragment --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out-dir fragments/

# 3. GFN2-xTB gaps on those fragments  (~1 s per COF, no ORCA)
python make_dft_features.py xtb-frags --frag-dir fragments/ --jobs 8

# 4. merge into the screener
python make_dft_features.py collect --frag-dir fragments/ --sacc sacc.csv \
       --screener ../dataset/pv_ranked_H.csv --out pv_ranked_H_gfn2.csv

# 5. the answer
python cof_pv_model.py analyze --gaps ../dataset/bandgap_CoRE_COF_v5.0_PCAdata.csv \
       --screener pv_ranked_H_gfn2.csv

'''

python make_dft_features.py sacc --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out sacc.csv
python make_dft_features.py collect --frag-dir fragments/ --sacc sacc.csv \
       --screener pv_ranked_H.csv --out pv_ranked_H_gfn2.csv
python cof_pv_model.py analyze --gaps ../dataset/bandgap_CoRE_COF_v5.0_PCAdata.csv \
       --screener pv_ranked_H_gfn2.csv