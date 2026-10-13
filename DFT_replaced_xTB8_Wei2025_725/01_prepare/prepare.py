#!/usr/bin/env python3
"""Prepare all Wei-725 COFs and reuse xTB5-identical CoRE fragment results."""
from __future__ import annotations
import csv, json
from pathlib import Path

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent; ROOT=PROJECT.parents[1]
WEI=ROOT/'wei2025_cof_bandgap_dataset'
XTB4=ROOT/'COF_Bandgap_Code&Data/DFT_replaced_xTB4'
XTB2=ROOT/'COF_Bandgap_Code&Data/DFT_replaced_xTB2'
XTB6=ROOT/'COF_Bandgap_Code&Data/DFT_replaced_xTB6_Wei2025_external'

rows=list(csv.DictReader((WEI/'unified_dataset.csv').open()))
train_ids={r['COF'] for r in csv.DictReader((XTB4/'01_feature_table/xtb4_features.csv').open())}
overlap=[r for r in rows if r['id_scheme']=='CoRE-COF positional ID' and r['source_id'] in train_ids]
new=[r for r in rows if r not in overlap]

staged=HERE/'all_cifs'; staged.mkdir(parents=True,exist_ok=True)
for r in rows:
 out=staged/r['filename']; source=WEI/'cifs'/r['filename']
 if not out.exists(): out.symlink_to(source.resolve())

# Reuse successful three-fragment directories produced for the 569 genuinely new
# CIFs. This avoids repeating an identical fragmentation calculation.
fragments=PROJECT/'02_fragments'; fragments.mkdir(parents=True,exist_ok=True)
for r in new:
 source=XTB6/'02_fragments'/Path(r['filename']).stem
 dest=fragments/Path(r['filename']).stem
 if source.is_dir() and all((source/f'{role}.xyz').is_file() for role in ('node','linker','nodelinker')) and not dest.exists():
  dest.symlink_to(source.resolve(), target_is_directory=True)

# Seed the rich-xTB cache for the 156 IDs already present in xTB5. Their CIF
# IDs refer to the same CoRE v5.0 positional structures; only filenames differ.
old_cache=json.loads((XTB2/'04_rich_xTB_extraction/rich_xtb_cache.json').read_text())
seed={}
for r in overlap:
 for role in ('node','linker','nodelinker'):
  value=old_cache.get(f"{r['source_id']}.cif:{role}")
  if value and 'error' not in value: seed[f"{r['filename']}:{role}"]=value

with (HERE/'wei725_screener.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['file']); w.writeheader(); w.writerows({'file':r['filename']} for r in rows)
with (HERE/'wei725_manifest.csv').open('w',newline='') as h:
 fields=list(rows[0])+['xtb5_id_overlap','rich_xtb_source']
 w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
 for r in rows:
  shared=r in overlap
  w.writerow({**r,'xtb5_id_overlap':'yes' if shared else 'no','rich_xtb_source':'reused_xtb5' if shared else 'new_calculation'})

out_cache=PROJECT/'03_rich_xTB'/'rich_xtb_cache.json'; out_cache.parent.mkdir(parents=True,exist_ok=True)
out_cache.write_text(json.dumps(seed,indent=2,allow_nan=True))
audit={'wei_labeled_rows':len(rows),'reused_identical_corecof_xtb_records':len(overlap),
       'new_wei_records_requiring_xtb':len(new),'new_records_with_complete_fragment_geometry':sum((XTB6/'02_fragments'/Path(r['filename']).stem/'node.xyz').is_file() and (XTB6/'02_fragments'/Path(r['filename']).stem/'linker.xyz').is_file() and (XTB6/'02_fragments'/Path(r['filename']).stem/'nodelinker.xyz').is_file() for r in new),
       'new_records_missing_one_or_more_fragment_geometry':sum(not ((XTB6/'02_fragments'/Path(r['filename']).stem/'node.xyz').is_file() and (XTB6/'02_fragments'/Path(r['filename']).stem/'linker.xyz').is_file() and (XTB6/'02_fragments'/Path(r['filename']).stem/'nodelinker.xyz').is_file()) for r in new),
       'seeded_rich_xtb_fragment_tasks':len(seed)}
(HERE/'cohort_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
