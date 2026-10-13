# xTB5: TabPFN V3 benchmark

This experiment changes only the TabPFN foundation-model version from V2 to
V3. It deliberately reuses xTB4's 216-COF feature table, 92 usable DFT-free
features, and five family-grouped folds, so its score can be compared directly
with xTB4 TabPFN V2.

Result: V3 reached R2 0.6446 and MAE 0.3808 eV, compared with V2 R2 0.6235 and
MAE 0.3814 eV on the same out-of-fold predictions.

```bash
TABPFN_DISABLE_TELEMETRY=1 ../../.venv312/bin/python 01_benchmark_v3/run.py
```
