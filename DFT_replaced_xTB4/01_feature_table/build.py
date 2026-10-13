#!/usr/bin/env python3
"""Build xTB4 features from the canonical 315-row DFT-free feature table."""
from __future__ import annotations
import csv, importlib.util, json
from collections import Counter
from pathlib import Path
import numpy as np
from pymatgen.core import Structure

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent; ROOT=PROJECT.parent
spec=importlib.util.spec_from_file_location('x2',ROOT/'DFT_replaced_xTB2/common.py')
x2=importlib.util.module_from_spec(spec);spec.loader.exec_module(x2)
rich_rows=list(csv.DictReader((ROOT/'DFT_replaced_xTB2/04_rich_xTB_extraction/rich_xtb_features.csv').open()))
rich={r['file'].replace('.cif',''):r for r in rich_rows}

# This is the authoritative source of COF targets and inexpensive framework
# descriptors.  It contains 315 COFs with a finite band-gap label.  The older
# PCA table is intentionally not used to construct xTB4's target column.
feature_table=ROOT/'dataset_code/bandgap_CoRE_COF_v5.0_DFTfree_features.csv'
with feature_table.open(encoding='utf-8-sig',newline='') as handle:
 dftfree={r['COF'].strip():r for r in csv.DictReader(handle)}

# Fragment SMILES/link family are needed only to assign family-grouped folds;
# they are not DFT-derived model inputs.
metadata={r['id']:r for r in x2.load_rows()}

def finite(v):
 try:return np.isfinite(float(v))
 except:return False

rows=[]
for cof,features in dftfree.items():
 meta=metadata.get(cof)
 if not meta or cof not in rich or rich[cof]['xtb_status']!='OK' or not finite(features.get('band_gap_eV')):
  continue
 # Use the DFT-free table values for every inexpensive framework descriptor.
 row={**meta, 'id':cof, 'gap':float(features['band_gap_eV']),
      'density':float(features['density_g_cm3']), 'volume':float(features['volume_A3']),
      'sacc':float(features['sacc_m2_cm3']), 'n_atoms':float(features['n_atoms']),
      'layered':float(features['layered']), 'stack_len':float(features['stack_len_A']),
      'n_pbc_bonds_cut':float(features['n_pbc_bonds_cut']),
      'n_unique_fragments':float(features['n_unique_fragments']),
      'volume_per_atom':float(features['volume_per_atom_A3']),
      'pbc_bond_per_atom':float(features['pbc_bond_density']),
      'n_count':float(features['n_count']), 'b_count':float(features['b_count']),
      'o_count':float(features['o_count'])}
 rows.append(row)

def num(v):
 try:return float(v)
 except:return np.nan

roles=['node','linker','nodelinker']
props=['homo_eV','lumo_eV','gap_eV','vertical_ip_eV','vertical_ea_eV',
 'electronegativity_eV','hardness_eV','dipole_debye','charge_mean_abs_e',
 'charge_std_e','charge_min_e','charge_max_e','bond_order_mean','bond_order_max',
 'bond_order_std','n_xtb_atoms']
rich_names=[f'{role}_{prop}' for role in roles for prop in props]
interaction_names=[
 'homo_node_minus_linker','lumo_node_minus_linker','ip_node_minus_linker',
 'ea_node_minus_linker','frontier_cross_gap','min_fragment_gap','max_fragment_gap',
 'nodelinker_gap_lowering','nodelinker_vs_fragment_mean','gap_ratio_node_linker',
 'dipole_change_pair_vs_sum','charge_polarization_node_linker',
 'bond_order_pair_minus_mean','size_ratio_node_linker']
structure_extra_names=['n_frac','b_frac','o_frac','hetero_frac','sacc_per_atom',
 'sacc_times_density','surface_volume_proxy','packing_inverse_volume_per_atom',
 'lattice_min_A','lattice_mid_A','lattice_max_A','lattice_anisotropy',
 'lattice_aspect_mid_min','alpha_deg','beta_deg','gamma_deg','angle_deviation_deg']

def val(record,role,prop):return num(record.get(f'{role}_xtb_{prop}'))
def interactions(record):
 n=lambda p:val(record,'node',p);l=lambda p:val(record,'linker',p);nl=lambda p:val(record,'nodelinker',p)
 ng,lg=n('gap_eV'),l('gap_eV')
 return [n('homo_eV')-l('homo_eV'),n('lumo_eV')-l('lumo_eV'),
  n('vertical_ip_eV')-l('vertical_ip_eV'),n('vertical_ea_eV')-l('vertical_ea_eV'),
  min(n('lumo_eV'),l('lumo_eV'))-max(n('homo_eV'),l('homo_eV')),
  min(ng,lg),max(ng,lg),nl('gap_eV')-min(ng,lg),nl('gap_eV')-(ng+lg)/2,
  ng/max(lg,1e-6),nl('dipole_debye')-(n('dipole_debye')+l('dipole_debye')),
  abs(n('charge_mean_abs_e')-l('charge_mean_abs_e')),
  nl('bond_order_mean')-(n('bond_order_mean')+l('bond_order_mean'))/2,
  n('n_xtb_atoms')/max(l('n_xtb_atoms'),1e-6)]

baseline_names=list(x2.STRUCTURE)+['node_gap','linker_gap','nodelinker_gap',
 'node_minus_linker_gap','min_gap','max_gap','nodelinker_minus_min_gap']
records=[]
for row in rows:
 rec=rich[row['id']]; ng,lg,nlg=[val(rec,r,'gap_eV') for r in roles]
 baseline=[row[x] for x in x2.STRUCTURE]+[ng,lg,nlg,ng-lg,min(ng,lg),max(ng,lg),nlg-min(ng,lg)]
 n_atoms=row['n_atoms']; hetero=row['n_count']+row['b_count']+row['o_count']
 cif=ROOT/'dataset_code/CoRE-COFs_DT591-v5.0'/f"{row['id']}.cif"
 try:
  st=Structure.from_file(cif); lengths=sorted(st.lattice.abc);angles=st.lattice.angles
 except Exception:
  lengths=[np.nan]*3;angles=[np.nan]*3
 extras=[row['n_count']/n_atoms,row['b_count']/n_atoms,row['o_count']/n_atoms,
  hetero/n_atoms,row['sacc']/n_atoms,row['sacc']*row['density'],
  row['sacc']/row['volume'],1/max(row['volume_per_atom'],1e-8),
  *lengths,lengths[2]/lengths[0],lengths[1]/lengths[0],*angles,
  np.mean(np.abs(np.asarray(angles)-90))]
 rich_values=[val(rec,role,prop) for role in roles for prop in props]
 values=baseline+rich_values+interactions(rec)+extras
 records.append({'COF':row['id'],'family':row['family'],'band_gap_eV':row['gap'],
                 **dict(zip(baseline_names+rich_names+interaction_names+structure_extra_names,values))})
fields=list(records[0]);out=HERE/'xtb4_features.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
(HERE/'feature_groups.json').write_text(json.dumps({'baseline':baseline_names,
 'physics_informed':baseline_names+rich_names+interaction_names+structure_extra_names},indent=2))
audit={'target_and_framework_feature_source':str(feature_table),
 'n_labeled_in_dftfree_table':int(sum(finite(r.get('band_gap_eV')) for r in dftfree.values())),
 'n_with_successful_node_linker_and_nodelinker_xtb':len(records),
 'n_excluded_for_missing_or_failed_rich_xtb':int(sum(finite(r.get('band_gap_eV')) for r in dftfree.values())-len(records)),
 'fragment_family_metadata_source':str(ROOT/'DFT_replaced_xTB2/dataset_code/pv_ranked_H.csv'),
 'rich_xtb_status_counts':dict(Counter(r['xtb_status'] for r in rich_rows)),
 'model_uses_dft_derived_inputs':False}
(HERE/'data_cohort_audit.json').write_text(json.dumps(audit,indent=2))
print('rows',len(records),'features',len(fields)-3,'wrote',out)
