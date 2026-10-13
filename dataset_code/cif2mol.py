import copy
from ase.io import read
from rdkit import Chem
from rdkit.Chem import RWMol
from utils_pbc import xyz2AC,AC2mol_PBC,transform_adj_matrix_PB
from utils import AC2mol
from periodic_smiels import mol_from_periodic_smiles


def get_central_cell_indices(supercell, lower_bound=1/3, upper_bound=2/3, tolerance=1e-16):
    """
    Identify indices of atoms within the 'central cell' of a supercell
    based on fractional (scaled) coordinates.
    
    Parameters
    ----------
    supercell : ase.Atoms
        The ASE Atoms object of the supercell.
    lower_bound : float, optional
        The lower bound in fractional coordinates for the central cell.
    upper_bound : float, optional
        The upper bound in fractional coordinates for the central cell.
    tolerance : float, optional
        A small tolerance for numerical inaccuracies.
    
    Returns
    -------
    list of int
        A list containing the indices of atoms belonging to the central cell.
    """
    frac_positions = supercell.get_scaled_positions()
    central_indices = []

    for i, frac_pos in enumerate(frac_positions):
        in_central_cell = True
        for coord in frac_pos:
            if not (lower_bound - tolerance <= coord < upper_bound - tolerance):
                in_central_cell = False
                break
        if in_central_cell:
            central_indices.append(i)

    return central_indices


def create_reverse_mapping(central_cell_indices, original_cell_size, supercell_size):
    """
    Create a reverse mapping from supercell atom indices to their
    corresponding 'central cell' atom indices.
    
    Parameters
    ----------
    central_cell_indices : list of int
        Indices of atoms in the central cell (relative to the supercell).
    original_cell_size : int
        Number of atoms in the original (primitive) cell.
    supercell_size : int
        Total number of atoms in the supercell.
    
    Returns
    -------
    dict
        A dictionary where keys are supercell atom indices and values
        are the corresponding indices in the central cell.
    """
    # Forward mapping: for each central index, gather all equivalent indices in the supercell
    mapping = {
        idx: list(range(idx % original_cell_size, supercell_size, original_cell_size))
        for idx in central_cell_indices
    }

    # Reverse mapping: supercell atom index -> central cell index
    reverse_mapping = {}
    for central_idx, cloned_indices in mapping.items():
        for cloned_idx in cloned_indices:
            reverse_mapping[cloned_idx] = central_idx

    return reverse_mapping


def build_rdkit_mol_from_supercell(
    atoms,
    repeat=(3, 3, 3),
    lower_bound=1/3,
    upper_bound=2/3,
    tolerance=1e-16,
    charge=0
):
    """
    Build an RDKit molecule representing just the central cell of a repeated supercell,
    accounting for periodic bonds.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Original ASE Atoms object (primitive cell).
    repeat : tuple of int, optional
        Repetition factors along the x, y, z directions.
    lower_bound : float, optional
        The lower fractional coordinate defining the central cell.
    upper_bound : float, optional
        The upper fractional coordinate defining the central cell.
    tolerance : float, optional
        A small tolerance for numeric inaccuracies in fractional coordinates.
    charge : int, optional
        Total charge of the system (if relevant).
    
    Returns
    -------
    rdkit.Chem.Mol
        An RDKit molecule object representing the central cell and its periodic bonds.
    """

    # 1. Create the 3x3x3 supercell
    supercell = atoms.repeat(repeat)

    # 2. Identify central cell indices
    central_indices = get_central_cell_indices(
        supercell, lower_bound=lower_bound, upper_bound=upper_bound, tolerance=tolerance
    )

    # 3. Create the reverse mapping (supercell -> central cell)
    N = len(atoms)        # original cell size
    M = len(supercell)    # supercell size
    reverse_mapping = create_reverse_mapping(central_indices, N, M)

    # 4. Obtain adjacency matrix & coordinates directly in memory
    atomic_nums = supercell.get_atomic_numbers()
    coords = supercell.get_positions()

    # xyz2AC and transform_adj_matrix_PB are assumed to be your helper functions
    adj, xyzs = xyz2AC(atomic_nums, coords, charge)
    adj_new, adj_PBC = transform_adj_matrix_PB(adj, central_indices, reverse_mapping)

    # 5. Build an RWMol from the entire supercell
    rw_mol = RWMol(xyzs)

    # 6. Remove all atoms not in the central cell (in descending index order)
    for idx in reversed(range(rw_mol.GetNumAtoms())):
        if idx not in central_indices:
            rw_mol.RemoveAtom(idx)

    # 7. Build the final molecule(s) from adjacency
    # AC2mol_PBC is assumed to be another helper function
    remaining_atomic_nums = [atom.GetAtomicNum() for atom in rw_mol.GetAtoms()]
    mols = AC2mol_PBC(
        rw_mol,
        adj_new,
        adj_PBC,
        remaining_atomic_nums,
        charge,
        allow_charged_fragments=False,
        use_graph=True,
        use_atom_maps=False
    )
    if not mols:
        raise ValueError("No molecules were returned from AC2mol_PBC.")

    mol_central = mols[0]

    # 8. Trick to restore aromaticity by round-tripping through SMILES
    smiles = Chem.MolToSmiles(mol_central, isomericSmiles=True, allHsExplicit=True)

    # 9. Identify all periodic bonds
    periodic_bond_indices = []
    for bond in mol_central.GetBonds():
        if bond.HasProp("isPeriodic") and bond.GetProp("isPeriodic") == "true":
            periodic_bond_indices.append(bond.GetIdx())

    # 10. Construct the final molecule from periodic SMILES
    #     'mol_from_periodic_smiles' is assumed to handle periodic bond data in the SMILES string
    periodic_smiles = f"{smiles}|{periodic_bond_indices}"
    final_mol = mol_from_periodic_smiles(periodic_smiles)

    # 11. Add H atoms (just as a final step, if needed)
    final_mol_with_h = Chem.AddHs(copy.copy(final_mol))

    return final_mol_with_h