#!/usr/bin/env python3
"""Join Zenodo PBE0 fragment labels to successful xTB2 calculations."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from common import DESC_NAMES, LABELS, XTB_CACHE, descriptors

cache = json.loads(XTB_CACHE.read_text())
labels = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
xtb_fields = sorted({k for v in cache.values() if "error" not in v for k in v})
rows = []
for label in labels:
    key = f"{label['COF']}.cif:{label['role']}"
    result = cache.get(key, {})
    if "error" in result or "xtb_gap_eV" not in result: continue
    row = {**label, **{k: result.get(k, np.nan) for k in xtb_fields}}
    row.update(dict(zip(DESC_NAMES, descriptors(label["canonical_smiles"]))))
    row["delta_gap_eV"] = float(label["pbe0_gap_eV"]) - float(result["xtb_gap_eV"])
    rows.append(row)

fields = list(rows[0])
out = HERE / "fragment_calibration_pairs.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
coverage = {"pbe0_label_occurrences": len(labels), "paired_xtb_pbe0": len(rows),
            "unique_fragments": len({r['canonical_smiles'] for r in rows}),
            "by_role": {role: sum(r['role']==role for r in rows)
                        for role in ('node','linker','nodelinker')}}
(HERE / "coverage.json").write_text(json.dumps(coverage, indent=2))
print(json.dumps(coverage, indent=2)); print(f"wrote {out}")
