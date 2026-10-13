#!/usr/bin/env python3
"""Run every experiment and write a compact comparison table."""
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scripts = [
    HERE / "00_structure_baseline" / "run.py",
    HERE / "01_rdkit_molecular" / "run.py",
    HERE / "02_xtb_electronic" / "run.py",
    HERE / "03_combined_residual" / "run.py",
]
for script in scripts:
    print(f"Running {script.parent.name} ...", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)

records = []
for metric_file in sorted(HERE.glob("*/results*/metrics.json")):
    data = json.loads(metric_file.read_text())
    for model, metrics in data["models"].items():
        records.append({"experiment": metric_file.parent.parent.name,
                        "result_set": metric_file.parent.name,
                        "label": data["label"], "model": model,
                        "n": data["n"], "n_families": data["n_families"],
                        **metrics})
with (HERE / "comparison.csv").open("w", newline="") as handle:
    fields = ["experiment", "result_set", "label", "model", "n",
              "n_families", "r2", "mae_eV"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(records)
print(f"Wrote {HERE / 'comparison.csv'}")

best = min(records, key=lambda row: float(row["mae_eV"]))
baseline = min((r for r in records if r["experiment"] == "00_structure_baseline"),
               key=lambda row: float(row["mae_eV"]))
summary = f"""# Experiment results

All scores are out-of-fold predictions from five-fold family-grouped CV.

## Best current result

- Experiment: `{best['experiment']}/{best['result_set']}`
- Model: `{best['model']}`
- Samples: {best['n']} ({best['n_families']} families)
- R2: {float(best['r2']):.3f}
- MAE: {float(best['mae_eV']):.3f} eV

## Common structural baseline

- Model: `{baseline['model']}`
- R2: {float(baseline['r2']):.3f}
- MAE: {float(baseline['mae_eV']):.3f} eV

See `comparison.csv` for every model/result set. Complete-case xTB results use
fewer samples than all-row results and should not be compared without noting
that population change. The all-row xTB experiment is the fair deployable
comparison against the 315-row structural baseline.
"""
(HERE / "RESULTS.md").write_text(summary)
