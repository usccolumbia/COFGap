#!/usr/bin/env python3
"""Stage a no-training-ID-overlap Wei-2025 external prediction cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
WEI = ROOT / "wei2025_cof_bandgap_dataset"
XTB4 = ROOT / "COF_Bandgap_Code&Data/DFT_replaced_xTB4"

with (WEI / "unified_dataset.csv").open(newline="") as handle:
    wei_rows = list(csv.DictReader(handle))
with (XTB4 / "01_feature_table/xtb4_features.csv").open(newline="") as handle:
    train_ids = {row["COF"] for row in csv.DictReader(handle)}

external = []
overlap = []
for row in wei_rows:
    is_core = row["id_scheme"] == "CoRE-COF positional ID"
    train_overlap = is_core and row["source_id"] in train_ids
    if train_overlap:
        overlap.append(row)
    else:
        external.append(row)

staged = HERE / "external_cifs"
staged.mkdir(exist_ok=True)
for row in external:
    source = WEI / "cifs" / row["filename"]
    destination = staged / row["filename"]
    if not source.is_file():
        raise FileNotFoundError(source)
    if not destination.exists():
        destination.symlink_to(source.resolve())

fields = list(wei_rows[0]) + ["external_evaluation_eligible", "overlaps_xtb5_training"]
with (HERE / "external_manifest.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in external:
        writer.writerow({**row, "external_evaluation_eligible": "yes", "overlaps_xtb5_training": "no"})
# The rich-xTB extractor needs only this filename column to locate fragments.
with (HERE / "external_screener.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["file"])
    writer.writeheader()
    writer.writerows({"file": row["filename"]} for row in external)
with (HERE / "training_overlap_excluded.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in overlap:
        writer.writerow({**row, "external_evaluation_eligible": "no", "overlaps_xtb5_training": "yes"})

audit = {
    "wei_total": len(wei_rows),
    "xtb5_training_cofs": len(train_ids),
    "excluded_existing_training_id_overlap": len(overlap),
    "external_cohort": len(external),
    "external_corecof": sum(r["id_scheme"] == "CoRE-COF positional ID" for r in external),
    "external_curatedcof": sum(r["id_scheme"] != "CoRE-COF positional ID" for r in external),
    "note": "Wei reference labels use a different reported target convention from xTB5 labels; do not interpret external label metrics as directly comparable until target alignment is established.",
}
(HERE / "cohort_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
