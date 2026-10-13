#!/usr/bin/env python
# coding: utf-8

# In[2]:


from rdkit.Chem.Draw import IPythonConsole
import rdkit
print(rdkit.__version__)
IPythonConsole.molSize = 1700,1700
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem




# In[3]:


IPythonConsole.drawOptions.addAtomIndices = False


# In[4]:


from mini_tools import highlight_shared_bonds
from chouse_bonds import create_graph_from_bonds,find_symmetrical_triangles_or_all,create_adjacency_matrix,frag_func,guilatine,get_fragments_with_unique_ids, combine
from chem_rules import Rule1, Rule2, Rule3, Rule4


# In[5]:


from cif2mol import build_rdkit_mol_from_supercell


# This code convert cif to mol graph

# In[6]:


icof = 447
# Read the primitive cell from a CIF file
atoms = read(f"{icof}.cif")

# Create the final molecule from the supercell
final_mol = build_rdkit_mol_from_supercell(atoms, repeat=(3, 3, 3))

# For debugging or confirmation
print("Final molecule has:", final_mol.GetNumAtoms(), "atoms.")


# In[7]:


mol_fragment = Chem.GetMolFrags(final_mol, asMols=True)
len_cof_frag=len(mol_fragment)
for i, single_mol in enumerate(mol_fragment):
    if len(mol_fragment[i].GetAtoms()) ==len(mol_fragment[0].GetAtoms()):
        single_mol = Chem.AddHs(single_mol)
        Chem.SanitizeMol(single_mol)
    else:
        single_mol=None
single_mol.Debug()


# In[8]:


rules = [Rule1(), Rule4(), Rule2(), Rule3()]


# In[9]:


if single_mol!=None:



    out_bonds = set()
    current_mol = single_mol
    for rule in rules:
        matched_bonds, current_mol = rule.apply(current_mol,
                                                highlight_shared_bonds(single_mol)[1],highlight_shared_bonds(single_mol)[1],highlight_shared_bonds(single_mol)[2])

        out_bonds.update(matched_bonds)

    #here are all the bonds that satisfy our rules
    suspected_bonds = list(out_bonds)  # removes duplicates

    #some additional variables for later use
    mol=current_mol 
    mol_out=current_mol 

    # Generate bond fingerprints
    radius = 2
    fingerprint_dict = {}  # Assuming you've initialized this dictionary before

    for bond_id in suspected_bonds:
        bond = mol.GetBondWithIdx(bond_id)
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()

        # Generate fingerprints centered around the two atoms of the bond
        fp_begin = AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=False,useBondTypes=False, fromAtoms=[begin_atom])
        fp_end = AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=False,useBondTypes=False, fromAtoms=[end_atom])

        # Combine the two fingerprints with a logical OR
        combined_fp = fp_begin | fp_end
        # Convert to a hashable representation
        hashable_fp = combined_fp.ToBitString()
        fingerprint_dict[bond_id] = hashable_fp

    # Group bonds with identical fingerprint representations
    grouped_bonds = {}
    for bond_id, fp in fingerprint_dict.items():
        if fp not in grouped_bonds:
            grouped_bonds[fp] = []
        grouped_bonds[fp].append(bond_id)

    # Convert to list of lists format
    final_groups = [group for group in grouped_bonds.values() if len(group) > 1]

    if len(final_groups) > 1:
        # Sort by the size of the group and then by fingerprint value
        sorted_groups = sorted(final_groups, key=lambda g: (-len(g), fingerprint_dict[g[0]]))

        # The first group will be the one with the highest count and the smallest fingerprint value
        suspected_bonds = sorted_groups[0]

    # Creating the graph
    bond_list = suspected_bonds
    G = create_graph_from_bonds(mol, bond_list)


    matrix, nodes_order = create_adjacency_matrix(G)
    #result = find_smallest_labels(matrix, nodes_order)
    triangles_or_all = find_symmetrical_triangles_or_all(matrix, nodes_order)
    set_bonds_fin=set()
    for i in triangles_or_all:
        set_bonds_fin.update(i)



# In[10]:


out=guilatine(mol_out,set_bonds_fin) # guilatine cuts the bonds
out_frag=Chem.GetMolFrags(out, asMols=True,sanitizeFrags=False)
fragment_list=frag_func(out_frag)


# In[11]:


if len(get_fragments_with_unique_ids(fragment_list)) ==2:

    out_1_m=get_fragments_with_unique_ids(fragment_list)[0]
    out_2_m=get_fragments_with_unique_ids(fragment_list)[1]
    out_sm1=Chem.MolToSmiles(out_1_m)
    out_sm2=Chem.MolToSmiles(out_2_m)
    ll=Chem.MolFromSmiles(out_sm1)
    ll2=Chem.MolFromSmiles(out_sm2)  
    ll.GetAtoms()
    ll2.GetAtoms()


    print(out_sm1)
    print(out_sm2)

    com=combine([out_1_m,out_2_m])
    com_smile=Chem.MolToSmiles(com)

    Chem.MolFromSmiles(com_smile)
    results = Chem.Draw.MolsToGridImage([out_1_m,out_2_m,com],maxMols=10,subImgSize=(700,700),legends=["amount:"+str(out_1_m.GetUnsignedProp("amount"))+", n_conncections:" + str(out_1_m.GetUnsignedProp("number_connections"))+", type" + str(out_1_m.GetUnsignedProp("type"))+", rule:" + str(out_1_m.GetUnsignedProp("rule")),"amount:"+str(out_2_m.GetUnsignedProp("amount"))+", n_conncections:" + str(out_2_m.GetUnsignedProp("number_connections"))+", type" + str(out_2_m.GetUnsignedProp("type"))+", rule:" + str(out_2_m.GetUnsignedProp("rule")), "n+l"],  returnPNG=False,molsPerRow=3)

    results.save("l_n_test/" + str(icof)+".png")
    text=str(icof)+" || "+"COFs_in_cell: "+str(len_cof_frag)+" || "+out_sm1+" ,"+"amount:"+str(out_1_m.GetUnsignedProp("amount"))+", n_conncections:" + str(out_1_m.GetUnsignedProp("number_connections"))+", type" + str(out_1_m.GetUnsignedProp("type"))+", rule:" + str(out_1_m.GetUnsignedProp("rule"))+" ||| "+ out_sm2+" ,""amount:"+str(out_2_m.GetUnsignedProp("amount"))+", n_conncections:" + str(out_2_m.GetUnsignedProp("number_connections"))+", type" + str(out_2_m.GetUnsignedProp("type"))+", rule:" + str(out_2_m.GetUnsignedProp("rule"))+" comb_smiles: " +com_smile

    print("+COF:"+str(icof))

elif len(get_fragments_with_unique_ids(out_frag)) >2:
    print("more then 2 fragments in the output")
    print("-COF:"+str(icof))


# In[ ]:




