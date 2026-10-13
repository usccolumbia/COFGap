# 1) compute xTB gaps  (~10 s/COF single-threaded; ~15 min at --jobs 8)
python compute_xtb_gaps.py --screener pv_ranked_H.csv \
    --out pv_ranked_H_xtb.csv --geometry cif \
    --cif-dir CoRE-COFs_DT591-v5.0/ --jobs 8

# 2) the actual test: does grouped R² move from 0.28 toward 0.57?
python cof_pv_model_xtb.py analyze --gaps bandgap_CoRE_COF_v5.0_PCAdata.csv \
    --screener pv_ranked_H_xtb.csv

# 3) only if it helps — save a deployable model that uses it
python cof_pv_model_xtb.py train --gaps bandgap_CoRE_COF_v5.0_PCAdata.csv \
    --screener pv_ranked_H_xtb.csv --features deployable+xtb \
    --model-out pv_model_xtb.joblib