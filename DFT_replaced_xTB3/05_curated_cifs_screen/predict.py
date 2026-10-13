#!/usr/bin/env python3
"""Assemble curated-CIF features and predict with packaged xTB3 model."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import joblib
import numpy as np


def number(value):
    try: return float(value)
    except (TypeError, ValueError): return np.nan


def keyed(path, key):
    return {r[key].replace('.cif',''):r for r in csv.DictReader(path.open())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--smiles',type=Path,required=True)
    p.add_argument('--sacc',type=Path,required=True)
    p.add_argument('--gaps',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); bundle=joblib.load(a.model)
    structures=keyed(a.smiles,'file'); sacc=keyed(a.sacc,'file'); gaps=keyed(a.gaps,'cof')
    results=[]
    for cof,row in structures.items():
        gaprow=gaps.get(cof,{})
        raw=[number(gaprow.get(k)) for k in ('node','linker','nodelinker')]
        missing=[k for k,v in zip(('node','linker','nodelinker'),raw) if not np.isfinite(v)]
        n=number(row.get('n_atoms')); volume=number(row.get('volume_A3'))
        pbc=number(row.get('n_pbc_bonds_cut'))
        features=[number(row.get('density_g_cm3')),volume,
                  number(sacc.get(cof,{}).get('sacc')),n,
                  1.0 if row.get('layered','').lower()=='yes' else 0.0,
                  number(row.get('stacking_len_A')),pbc,
                  number(row.get('n_unique_fragments')),
                  volume/n if n>0 else np.nan,pbc/n if n>0 else np.nan]
        if missing:
            results.append({'file':cof+'.cif','status':'MISSING_XTB:'+','.join(missing),
                            'pred_gap_eV':'','in_PV_window':''}); continue
        node,linker,nl=raw
        features += [node,linker,nl,node-linker,min(node,linker),max(node,linker),nl-min(node,linker)]
        pred=float(bundle['model'].predict(np.asarray(features).reshape(1,-1))[0])
        results.append({'file':cof+'.cif','status':'OK','pred_gap_eV':pred,
                        'in_PV_window':'yes' if 1.5<=pred<=2.5 else 'no',
                        'node_xtb_gap_eV':node,'linker_xtb_gap_eV':linker,
                        'nodelinker_xtb_gap_eV':nl})
    results.sort(key=lambda r:(r['status']!='OK',abs(number(r.get('pred_gap_eV'))-2.0)))
    fields=['file','status','pred_gap_eV','in_PV_window','node_xtb_gap_eV',
            'linker_xtb_gap_eV','nodelinker_xtb_gap_eV']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    ok=[r for r in results if r['status']=='OK']; pv=[r for r in ok if r['in_PV_window']=='yes']
    with (a.out.parent/'PV_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pv)
    summary={'total':len(results),'predicted':len(ok),'failed':len(results)-len(ok),
             'pv_window_1p5_2p5_eV':len(pv),
             'predicted_gap_min_eV':min((r['pred_gap_eV'] for r in ok),default=None),
             'predicted_gap_max_eV':max((r['pred_gap_eV'] for r in ok),default=None)}
    (a.out.parent/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print('top candidates')
    for r in pv[:15]:print(r['file'],f"{r['pred_gap_eV']:.3f} eV")

if __name__=='__main__':main()
