#!/usr/bin/env python3
"""Isolated worker: CIF -> H-capped molecule -> rich GFN2-xTB properties."""
from __future__ import annotations
import json
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
HARTREE_EV = 27.211386245988
EBOHR_DEBYE = 2.541746473


def calculator(numbers, positions, charge=0):
    from tblite.interface import Calculator
    electrons = int(numbers.sum() - charge)
    calc = Calculator("GFN2-xTB", numbers, positions,
                      charge=float(charge), uhf=int(electrons % 2))
    calc.set("verbosity", 0)
    calc.set("max-iter", 400)
    if electrons % 2:
        calc.add("spin-polarization")
    return calc


def main():
    source, package_dir = sys.argv[1:3]
    if source.lower().endswith(".xyz"):
        from pymatgen.core import Molecule
        molecule = Molecule.from_file(source)
        numbers = np.asarray(molecule.atomic_numbers, dtype=int)
        xyz = np.asarray(molecule.cart_coords, dtype=float) / 0.529177210903
    else:
        sys.path.insert(0, package_dir)
        from rdkit import RDLogger
        from pymatgen.core import Structure
        import cof_cif2smiles as converter
        RDLogger.DisableLog("rdApp.*")
        structure = Structure.from_file(source)
        graph = converter.prune_overbonded(structure,
            converter.build_quotient_graph(structure))
        positions, internal, periodic = converter.unwrap(structure, graph)
        molecule = converter.to_rdkit(structure, positions, internal, periodic,
                                      cap="H").GetMol()
        conformer = molecule.GetConformer()
        numbers = np.array([a.GetAtomicNum() for a in molecule.GetAtoms()], dtype=int)
        xyz = np.array([list(conformer.GetAtomPosition(i))
                        for i in range(molecule.GetNumAtoms())]) / 0.529177210903
    if not len(numbers):
        raise ValueError("empty molecule")

    neutral_calc = calculator(numbers, xyz, 0)
    neutral_calc.add("molecular-multipoles")
    neutral_calc.add("bond-orders")
    neutral = neutral_calc.singlepoint()
    orbital_e = np.asarray(neutral.get("orbital-energies"), dtype=float)
    occupations = np.asarray(neutral.get("orbital-occupations"), dtype=float)
    if orbital_e.ndim > 1:
        orbital_e = orbital_e.reshape(-1)
        occupations = occupations.reshape(-1)
    occupied = np.sort(orbital_e[occupations > 0.5]) * HARTREE_EV
    virtual = np.sort(orbital_e[occupations <= 0.5]) * HARTREE_EV
    homo, lumo = occupied[-1], virtual[0]

    energy0 = float(neutral.get("energy"))
    try:
        energy_cation = float(calculator(numbers, xyz, +1).singlepoint().get("energy"))
        ip = (energy_cation - energy0) * HARTREE_EV
    except Exception:
        ip = np.nan
    try:
        energy_anion = float(calculator(numbers, xyz, -1).singlepoint().get("energy"))
        ea = (energy0 - energy_anion) * HARTREE_EV
    except Exception:
        ea = np.nan
    hardness = max((ip - ea) / 2.0, 1e-8) if np.isfinite(ip + ea) else np.nan
    electronegativity = (ip + ea) / 2.0

    charges = np.asarray(neutral.get("charges"), dtype=float)
    dipole = np.asarray(neutral.get("dipole"), dtype=float) * EBOHR_DEBYE
    quadrupole = np.asarray(neutral.get("quadrupole"), dtype=float)
    bond_orders = np.asarray(neutral.get("bond-orders"), dtype=float)
    upper = bond_orders[np.triu_indices(len(numbers), 1)]
    bonded = upper[upper > 0.05]
    result = {
        "n_xtb_atoms": int(len(numbers)),
        "xtb_total_energy_hartree": energy0,
        "xtb_homo_eV": float(homo),
        "xtb_homo_minus_1_eV": float(occupied[-2] if len(occupied) > 1 else np.nan),
        "xtb_lumo_eV": float(lumo),
        "xtb_lumo_plus_1_eV": float(virtual[1] if len(virtual) > 1 else np.nan),
        "xtb_gap_eV": float(lumo - homo),
        "xtb_vertical_ip_eV": float(ip),
        "xtb_vertical_ea_eV": float(ea),
        "xtb_electronegativity_eV": float(electronegativity),
        "xtb_hardness_eV": float(hardness),
        "xtb_softness_per_eV": float(1.0 / (2.0 * hardness)) if np.isfinite(hardness) else np.nan,
        "xtb_electrophilicity_eV": float(electronegativity**2 / (2.0 * hardness)) if np.isfinite(hardness) else np.nan,
        "xtb_dipole_debye": float(np.linalg.norm(dipole)),
        "xtb_dipole_x_debye": float(dipole[0]),
        "xtb_dipole_y_debye": float(dipole[1]),
        "xtb_dipole_z_debye": float(dipole[2]),
        "xtb_quadrupole_norm_e_bohr2": float(np.linalg.norm(quadrupole)),
        "xtb_charge_min_e": float(charges.min()),
        "xtb_charge_max_e": float(charges.max()),
        "xtb_charge_mean_abs_e": float(np.mean(np.abs(charges))),
        "xtb_charge_std_e": float(charges.std()),
        "xtb_positive_charge_sum_e": float(charges[charges > 0].sum()),
        "xtb_negative_charge_sum_e": float(charges[charges < 0].sum()),
        "xtb_bond_order_mean": float(bonded.mean() if len(bonded) else 0.0),
        "xtb_bond_order_max": float(bonded.max() if len(bonded) else 0.0),
        "xtb_bond_order_std": float(bonded.std() if len(bonded) else 0.0),
        "xtb_bond_order_count": int(len(bonded)),
    }
    for symbol in ("B", "C", "N", "O", "S"):
        z = {"B": 5, "C": 6, "N": 7, "O": 8, "S": 16}[symbol]
        subset = charges[numbers == z]
        result[f"xtb_{symbol}_charge_mean_e"] = float(subset.mean()) if len(subset) else np.nan
        result[f"xtb_{symbol}_charge_std_e"] = float(subset.std()) if len(subset) else np.nan
    print(json.dumps(result, allow_nan=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
