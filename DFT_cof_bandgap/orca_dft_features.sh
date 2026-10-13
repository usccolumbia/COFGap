# one-time, this terminal session
export PATH="~/Downloads/project/2026-COF-jeffreyHu/bandgap/DFT_cof_bandgap/ORCA/orca_6_1_1_macosx_arm64_openmpi411:$PATH"

# 0. sanity check + surface area
python make_dft_features.py check
pip install pyzeo
python make_dft_features.py sacc --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out sacc.csv

# 1. fragment
python make_dft_features.py fragment --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out-dir fragments/
head -20 fragments/manifest.csv

# 2. pilot: 20 COFs, time it
mkdir -p pilot && ls -d fragments/*/ | head -20 | xargs -I{} cp -r {} pilot/
python make_dft_features.py orca-inputs --frag-dir pilot/ --level pbe0sp --nprocs 8 --tags nodelinker
time python make_dft_features.py orca-run --frag-dir pilot/ --jobs 2 --tags nodelinker

# 3. full run
python make_dft_features.py orca-inputs --frag-dir fragments/ --level pbe0sp --nprocs 8 --tags nodelinker
python make_dft_features.py orca-run --frag-dir fragments/ --jobs 4 --tags nodelinker

# 4. merge + retrain
python make_dft_features.py collect --frag-dir fragments/ --sacc sacc.csv \
       --screener pv_ranked_H.csv --out pv_ranked_H_orca.csv
python cof_pv_model.py analyze --gaps ../dataset/bandgap_CoRE_COF_v5.0_PCAdata.csv \
       --screener pv_ranked_H_orca.csv
       
# # 0. sanity check + surface area (fast, no DFT)
# export PATH="~/Downloads/project/2026-COF-jeffreyHu/bandgap/DFT_cof_bandgap/ORCA/orca_6_1_1_macosx_arm64_openmpi411:$PATH"

# python make_dft_features.py check
# # pip install pyzeo
# python make_dft_features.py sacc --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out sacc.csv

# # 1. fragment
# python make_dft_features.py fragment --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out-dir fragments/
# head -20 fragments/manifest.csv        # check node+linker sizes before committing

# # 2. PILOT: 20 COFs first, to time it
# mkdir -p pilot && ls -d fragments/*/ | head -20 | xargs -I{} cp -r {} pilot/
# python make_dft_features.py orca-inputs --frag-dir pilot/ --level pbe0sp \
#        --nprocs 8 --tags nodelinker
# time python make_dft_features.py orca-run --frag-dir pilot/ --jobs 2 --tags nodelinker

# # 3. full run (extrapolate the pilot timing first!)
# python make_dft_features.py orca-inputs --frag-dir fragments/ --level pbe0sp \
#        --nprocs 8 --tags nodelinker
# python make_dft_features.py orca-run --frag-dir fragments/ --jobs 4 --tags nodelinker

# # 4. merge + retrain
# python make_dft_features.py collect --frag-dir fragments/ --sacc sacc.csv \
#        --screener pv_ranked_H.csv --out pv_ranked_H_orca.csv
# python cof_pv_model.py analyze --gaps ../dataset/bandgap_CoRE_COF_v5.0_PCAdata.csv \
#        --screener pv_ranked_H_orca.csv