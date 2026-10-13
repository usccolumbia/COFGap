
import sys, numpy as np
sys.path.insert(0, sys.argv[2])          # dir containing cof_cif2smiles.py
cif = sys.argv[1]
backend = sys.argv[3] if len(sys.argv) > 3 else "tblite"
try:
    import warnings; warnings.filterwarnings("ignore")
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    from pymatgen.core import Structure
    import cof_cif2smiles as C
    s = Structure.from_file(cif)
    G = C.build_quotient_graph(s)
    G = C.prune_overbonded(s, G)
    pos, internal, periodic = C.unwrap(s, G)
    mol = C.to_rdkit(s, pos, internal, periodic, cap="H")   # H-capped, 3D
    m = mol.GetMol()
    conf = m.GetConformer()
    nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    xyz = np.array([list(conf.GetAtomPosition(i))
                    for i in range(m.GetNumAtoms())])
    if len(nums) == 0:
        print("FAIL empty"); sys.exit(0)
    if int(nums.sum()) % 2 != 0:
        print("FAIL odd_electrons"); sys.exit(0)
    p = xyz / 0.529177210903
    if backend == "tblite":
        from tblite.interface import Calculator
        calc = Calculator("GFN2-xTB", nums, p)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        e = np.array(res.get("orbital-energies")) * 27.211386
        occ = np.array(res.get("orbital-occupations"))
    else:
        from xtb.interface import Calculator as XCalc, Param
        calc = XCalc(Param.GFN2xTB, nums, p)
        res = calc.singlepoint()
        e = np.array(res.get_orbital_eigenvalues()) * 27.211386
        occ = np.array(res.get_orbital_occupations())
    homo = e[occ > 0.5].max(); lumo = e[occ <= 0.5].min()
    print(f"{lumo - homo:.6f}")
except Exception as exc:
    print("FAIL " + type(exc).__name__)
