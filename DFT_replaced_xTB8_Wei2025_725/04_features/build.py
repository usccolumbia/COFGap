#!/usr/bin/env python3
"""Build xTB4-compatible DFT-free features for complete Wei-725 xTB cases."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;ROOT=PROJECT.parents[1]
RICH=PROJECT/'03_rich_xTB'/'rich_xtb_features.csv';WEI=ROOT/'wei2025_cof_bandgap_dataset'
STRUCT=PROJECT/'04_features'/'structure_features.csv'
rich={r['file']:r for r in csv.DictReader(RICH.open())}
structure={r['file']:r for r in csv.DictReader(STRUCT.open())}
manifest=list(csv.DictReader((PROJECT/'01_prepare'/'wei725_manifest.csv').open()))
def num(x):
 try:return float(x)
 except:return np.nan
roles=['node','linker','nodelinker']
props=['homo_eV','lumo_eV','gap_eV','vertical_ip_eV','vertical_ea_eV','electronegativity_eV','hardness_eV','dipole_debye','charge_mean_abs_e','charge_std_e','charge_min_e','charge_max_e','bond_order_mean','bond_order_max','bond_order_std','n_xtb_atoms']
names=['density','volume','sacc','n_atoms','layered','stack_len','n_pbc_bonds_cut','n_unique_fragments','volume_per_atom','pbc_bond_per_atom','node_gap','linker_gap','nodelinker_gap','node_minus_linker_gap','min_gap','max_gap','nodelinker_minus_min_gap']
names += [f'{r}_{p}' for r in roles for p in props]
names += ['homo_node_minus_linker','lumo_node_minus_linker','ip_node_minus_linker','ea_node_minus_linker','frontier_cross_gap','min_fragment_gap','max_fragment_gap','nodelinker_gap_lowering','nodelinker_vs_fragment_mean','gap_ratio_node_linker','dipole_change_pair_vs_sum','charge_polarization_node_linker','bond_order_pair_minus_mean','size_ratio_node_linker']
names += ['n_frac','b_frac','o_frac','hetero_frac','sacc_per_atom','sacc_times_density','surface_volume_proxy','packing_inverse_volume_per_atom','lattice_min_A','lattice_mid_A','lattice_max_A','lattice_anisotropy','lattice_aspect_mid_min','alpha_deg','beta_deg','gamma_deg','angle_deviation_deg']
def val(r,role,prop):return num(r.get(f'{role}_xtb_{prop}'))
out=[]
for m in manifest:
 r=rich.get(m['filename']);s=structure.get(m['filename'])
 if not r or not s or r['xtb_status']!='OK' or s['status']!='OK':continue
 n,l,nl=(val(r,x,'gap_eV') for x in roles)
 nat=num(s['n_atoms']);vol=num(s['volume_A3']);dens=num(s['density_g_cm3']);pbc=num(s['n_pbc_bonds_cut'])
 cif=WEI/'cifs'/m['filename']
 try:
  st=Structure.from_file(cif); ls=sorted(st.lattice.abc);ang=st.lattice.angles; comp=st.composition
  counts={e:float(comp[e]) for e in ('N','B','O') if e in comp}; nc=counts.get('N',0);bc=counts.get('B',0);oc=counts.get('O',0)
 except: ls=[np.nan]*3;ang=[np.nan]*3;nc=bc=oc=np.nan
 layered=1.0 if str(s['layered']).lower()=='yes' else 0.0
 base=[dens,vol,np.nan,nat,layered,num(s['stacking_len_A']),pbc,2,vol/nat if nat else np.nan,pbc/nat if nat else np.nan,n,l,nl,n-l,min(n,l),max(n,l),nl-min(n,l)]
 richv=[val(r,role,p) for role in roles for p in props]
 N=lambda p:val(r,'node',p);L=lambda p:val(r,'linker',p);NL=lambda p:val(r,'nodelinker',p)
 inter=[N('homo_eV')-L('homo_eV'),N('lumo_eV')-L('lumo_eV'),N('vertical_ip_eV')-L('vertical_ip_eV'),N('vertical_ea_eV')-L('vertical_ea_eV'),min(N('lumo_eV'),L('lumo_eV'))-max(N('homo_eV'),L('homo_eV')),min(n,l),max(n,l),nl-min(n,l),nl-(n+l)/2,n/max(l,1e-6),NL('dipole_debye')-(N('dipole_debye')+L('dipole_debye')),abs(N('charge_mean_abs_e')-L('charge_mean_abs_e')),NL('bond_order_mean')-(N('bond_order_mean')+L('bond_order_mean'))/2,N('n_xtb_atoms')/max(L('n_xtb_atoms'),1e-6)]
 extras=[nc/nat,bc/nat,oc/nat,(nc+bc+oc)/nat,np.nan,np.nan,np.nan,1/max(vol/nat,1e-8),*ls,ls[2]/ls[0],ls[1]/ls[0],*ang,np.mean(np.abs(np.asarray(ang)-90))]
 # Family proxy uses source collection and capped elemental composition. The
 # Wei release lacks original linker-class metadata required by xTB5's family definition.
 family=f"{m['source_database']}|B{int(bc>0)}|N{int(min(nc,6))}|O{int(min(oc,6))}"
 out.append({'filename':m['filename'],'family_proxy':family,'reference_bandgap_eV':num(m['reference_bandgap_eV']),**dict(zip(names,base+richv+inter+extras))})
fields=list(out[0]);
with (HERE/'wei725_xtb_features.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(out)
(HERE/'feature_names.json').write_text(json.dumps(names,indent=2))
(HERE/'coverage.json').write_text(json.dumps({'wei_labeled_rows':len(manifest),'complete_rich_xtb_and_structure_rows':len(out),'excluded_rows':len(manifest)-len(out),'feature_count_requested':len(names)},indent=2))
print(len(out),'complete rows')
