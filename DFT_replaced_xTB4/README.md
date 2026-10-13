# DFT_replaced_xTB4

xTB4 compares the xTB3 random-forest baseline with pinned TabPFN v2 and
physics-informed features. No DFT-derived feature is used by the DFT-free
models. The original DFT gaps remain only as supervised targets/reference.

TabPFN package: 8.3.0. Model weights: `ModelVersion.V2`, selected because the
newer 2.6 weights require personal non-commercial-license authentication.

```bash
../../.venv312/bin/python 01_feature_table/build.py
TABPFN_DISABLE_TELEMETRY=1 ../../.venv312/bin/python 02_grouped_benchmark/run.py
TABPFN_DISABLE_TELEMETRY=1 ../../.venv312/bin/python 03_package_model/run.py
TABPFN_DISABLE_TELEMETRY=1 ../../.venv312/bin/python 03_package_model/predict.py \
  01_feature_table/xtb4_features.csv predictions.csv
```

All headline results use the same 216 COFs and five family-grouped folds as
xTB3. Feature-set comparison is performed before model comparison so gains
from representation and gains from TabPFN remain distinguishable.

The packaged predictor is a 75% TabPFN/25% random-forest blend. Its input CSV
must contain the 92 nonempty xTB4 source-feature columns; extra columns are
preserved in the output. The blend weight is a promising exploratory choice
from the current cross-validation predictions and should be locked before an
external or nested-CV confirmation.
