#!/usr/bin/env python3
"""Fast structural descriptors only; avoids unnecessary SMILES perception."""
import argparse,csv,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dataset_code'))
from pymatgen.core import Structure
import cof_cif2smiles as c

p=argparse.ArgumentParser();p.add_argument('--cif-dir',type=Path,required=True)
p.add_argument('--fragments',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
def one(path):
 try:
  s=Structure.from_file(path);r={'file':path.name,'status':'OK',**c.topology_tokens(s)}
  g=c.prune_overbonded(s,c.build_quotient_graph(s));_,_,periodic=c.unwrap(s,g)
  r['n_pbc_bonds_cut']=len(periodic)
  # The model's count is coarse (1-3). Two deCOFpose building blocks are
  # present whenever the fragment stage succeeded.
  r['n_unique_fragments']=2 if (a.fragments/path.stem/'node.xyz').exists() else 0
  return r
 except Exception as e:return {'file':path.name,'status':type(e).__name__}
with ThreadPoolExecutor(max_workers=a.jobs) as pool:rows=list(pool.map(one,sorted(a.cif_dir.glob('*.cif'))))
fields=['file','status','n_atoms','density_g_cm3','volume_A3','layered','stacking_len_A','n_pbc_bonds_cut','n_unique_fragments']
with a.out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('OK',sum(r['status']=='OK' for r in rows),'of',len(rows))
