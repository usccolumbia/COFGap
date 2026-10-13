# 1. fragments you already know are correct (node/linker/nodelinker XYZ)
#python make_dft_features.py fragment --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out-dir fragments/

# pip install pyzeo
python make_dft_features.py sacc --cif-dir ../dataset/CoRE-COFs_DT591-v5.0/ --out sacc.csv
# 2. xTB on the SAME node+linker construct, via ORCA's GFN2 driver
python make_dft_features.py orca-inputs --frag-dir fragments/ --level gfn2 --nprocs 1
python make_dft_features.py orca-run   --frag-dir fragments/ --jobs 8
python make_dft_features.py collect    --frag-dir fragments/ --sacc sacc.csv \
       --screener ../dataset/pv_ranked_H.csv --out pv_ranked_H_gfn2.csv