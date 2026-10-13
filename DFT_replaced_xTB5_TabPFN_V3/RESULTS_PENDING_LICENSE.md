# TabPFN V3 benchmark status

`ModelVersion.V3` is exposed by the installed `tabpfn==8.3.0` package.
The same-fold xTB4 V2-versus-V3 benchmark is implemented in
`01_benchmark_v3/run.py`, but V3 model-weight download requires a one-time
Prior Labs license acceptance and `TABPFN_TOKEN`.

The run was attempted on 2026-08-15. It successfully completed the V2 portion
but stopped when V3 requested its gated weights. No V3 R2 or MAE is reported
until the model is actually available and all five folds complete.

To authorize a rerun without exposing a token in project files, accept the
license at https://ux.priorlabs.ai, create an API key, and configure
`TABPFN_TOKEN` in the terminal/session that will launch the benchmark. Then
rerun:

```bash
TABPFN_MODEL_CACHE_DIR='../DFT_replaced_xTB4/tabpfn_models' \
TABPFN_DISABLE_TELEMETRY=1 \
../../.venv312/bin/python 01_benchmark_v3/run.py
```

The experiment reuses xTB4's 216 COFs, 92 usable DFT-free features, and the
same five family-grouped folds. Therefore the resulting V3 score will be a
direct comparison with V2 (R2 0.624; MAE 0.381 eV).
