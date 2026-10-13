#!/usr/bin/env python3
"""
cof_cif2smiles.py
=================
Convert periodic COF CIFs into SMILES strings for sequence models
(transformers) predicting band gaps.

A COF is an infinite covalent network, so it has no exact molecular SMILES.
This script produces, per CIF, up to three complementary representations:

  1. unit_smiles   - the full unit-cell contents unwrapped into one finite
                     molecule; bonds that cross into neighbouring cells are
                     cut and capped. Capping is either [*] wildcards
                     (connection points preserved, recommended for
                     transformers) or H atoms (chemically valid molecules).
  2. frag_smiles   - the unique building-block fragments obtained by cutting
                     the same PBC-crossing bonds AND splitting the remaining
                     graph at nothing else (i.e., connected components of the
                     capped unit) - deduplicated by canonical SMILES.
  3. aux tokens    - topology hints (density, atoms/cell, interlayer distance,
                     stacking-axis length) exported as extra CSV columns, so
                     you can append them to the sequence, e.g.:
                       <SMILES> | DENS_0.42 | NAT_180 | ILD_3.3
                     Papers show fragment chemistry alone explains only
                     ~R2 0.2 of COF band gaps - feed the model topology too.

Method
------
- Periodic bond graph from covalent radii (pymatgen CutOffDictNN-free,
  simple r_i + r_j + tol criterion, tol = 0.45 A).
- A spanning tree over the quotient graph unwraps each atom exactly once
  into consistent Cartesian coordinates; every non-tree bond whose image
  shift is non-zero is a "periodic" bond -> cut & capped.
- Bond orders perceived with RDKit rdDetermineBonds on the H-capped
  geometry; falls back to single-bond + aromatic ring perception if that
  fails (common for exotic valences).

Usage:
    python cof_cif2smiles.py <cif_dir_or_file> [--out cof_smiles.csv]
                             [--cap star|H] [--max-atoms 2000]

Dependencies: pymatgen, rdkit, networkx, numpy
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import networkx as nx

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

BOND_TOL = 0.45  # Angstrom added to sum of covalent radii
H_CAP_LEN = {"C": 1.09, "N": 1.01, "O": 0.96, "B": 1.19, "S": 1.34,
             "Si": 1.48, "P": 1.42}  # X-H bond lengths for capping


# ----------------------------------------------------------------------
# Periodic bond graph
# ----------------------------------------------------------------------
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius


def covalent_radius(sym):
    try:
        return float(CovalentRadius.radius[sym])
    except Exception:
        try:
            return float(Element(sym).atomic_radius)
        except Exception:
            return 0.8


def build_quotient_graph(struct: Structure):
    """
    Multigraph: nodes = atoms in the unit cell; edges carry the periodic
    image shift (jimage) of the neighbour. Uses covalent-radius criterion.
    """
    n = len(struct)
    radii = [covalent_radius(s.specie.symbol) for s in struct]
    cutoff = 2 * max(radii) + BOND_TOL
    centers, points, offsets, dists = struct.get_neighbor_list(r=cutoff)

    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    seen = set()
    for i, j, off, d in zip(centers, points, offsets, dists):
        if d < 0.5:  # same atom / duplicate site
            continue
        if d > radii[i] + radii[j] + BOND_TOL:
            continue
        key = (min(i, j), max(i, j), tuple(off if i < j else -off))
        if key in seen:
            continue
        seen.add(key)
        G.add_edge(int(i), int(j), jimage=tuple(int(x) for x in off))
    return G


MAX_BONDS = {"H": 1, "C": 4, "N": 4, "O": 2, "B": 4, "F": 1, "Cl": 1,
             "Br": 1, "I": 1, "S": 4, "Si": 4, "P": 5, "Se": 2}


def prune_overbonded(struct, G):
    """
    Real CIFs (disorder, close contacts) create too many bonds. Enforce a
    per-element maximum by dropping the LONGEST bonds on over-coordinated
    atoms until within limits.
    """
    pos = struct.cart_coords
    latt = struct.lattice

    def bond_len(u, v, jimage):
        d = pos[v] + np.array(jimage) @ latt.matrix - pos[u]
        return float(np.linalg.norm(d))

    changed = True
    while changed:
        changed = False
        for i in list(G.nodes):
            sym = struct[i].specie.symbol
            cap = MAX_BONDS.get(sym, 6)
            edges = [(u, v, k, d) for u, v, k, d in G.edges(i, keys=True,
                                                            data=True)]
            if len(edges) <= cap:
                continue
            edges.sort(key=lambda e: bond_len(e[0], e[1], e[3]["jimage"]))
            for u, v, k, d in edges[cap:]:
                if G.has_edge(u, v, k):
                    G.remove_edge(u, v, k)
                    changed = True
    return G


def unwrap(struct: Structure, G: nx.MultiGraph):
    """
    Assign each atom one Cartesian position, minimizing the number of cut
    (periodic) bonds. Two-phase: (1) grow islands over zero-image edges
    only, (2) connect islands via a spanning tree over the remaining
    edges. Chord edges whose accumulated image shift is non-zero are the
    genuine periodic bonds to cut.
    """
    latt = struct.lattice.matrix
    frac = struct.frac_coords
    n = len(struct)
    img = {i: None for i in range(n)}

    # Phase 1: flood-fill through jimage == (0,0,0) edges
    G0 = nx.Graph()
    G0.add_nodes_from(range(n))
    for u, v, d in G.edges(data=True):
        if d["jimage"] == (0, 0, 0):
            G0.add_edge(u, v)
    islands = list(nx.connected_components(G0))
    isl_of = {}
    for k, comp in enumerate(islands):
        for a in comp:
            isl_of[a] = k
    for comp in islands:
        root = min(comp)
        img[root] = np.zeros(3, int)
        for u, v in nx.bfs_edges(G0, root):
            img[v] = img[u]          # zero-image edge: same image

    # Phase 2: island graph over non-zero-image edges; BFS assigns island
    # offsets so tree edges become internal.
    IG = nx.Graph()
    IG.add_nodes_from(range(len(islands)))
    for u, v, d in G.edges(data=True):
        sh = np.array(d["jimage"], int)
        if not sh.any():
            continue
        a, b = isl_of[u], isl_of[v]
        if a != b and not IG.has_edge(a, b):
            IG.add_edge(a, b, u=u, v=v, shift=sh)
    isl_off = {k: None for k in range(len(islands))}
    for comp in nx.connected_components(IG):
        root = min(comp)
        isl_off[root] = np.zeros(3, int)
        for a, b in nx.bfs_edges(IG, root):
            e = IG[a][b]
            u, v, sh = e["u"], e["v"], e["shift"]
            if isl_of[u] == a:       # edge stored u->v with shift jimage
                isl_off[b] = isl_off[a] + (img[u] + sh) - img[v]
            else:
                isl_off[b] = isl_off[a] + (img[v] - sh) - img[u]
    for k in range(len(islands)):
        if isl_off[k] is None:
            isl_off[k] = np.zeros(3, int)
    for i in range(n):
        img[i] = img[i] + isl_off[isl_of[i]]

    pos = np.zeros((n, 3))
    for i in range(n):
        pos[i] = (frac[i] + img[i]) @ latt

    internal_bonds, periodic_bonds = [], []
    for u, v, d in G.edges(data=True):
        sh = np.array(d["jimage"], int)
        eff = img[u] + sh - img[v]
        if np.any(eff != 0):
            periodic_bonds.append((u, v, eff))
        else:
            internal_bonds.append((u, v))
    return pos, internal_bonds, periodic_bonds


# ----------------------------------------------------------------------
# RDKit molecule construction
# ----------------------------------------------------------------------
def to_rdkit(struct, pos, internal_bonds, periodic_bonds, cap="star"):
    """
    Build an RWMol: unit-cell atoms + caps replacing periodic bonds.
    Each periodic bond (u,v,shift) is replaced by TWO caps: one on u
    (pointing to v's image) and one on v (pointing to u's image), so
    every connection point is represented.
    """
    latt = None
    mol = Chem.RWMol()
    conf_pos = []
    for i, site in enumerate(struct):
        a = Chem.Atom(site.specie.symbol)
        mol.AddAtom(a)
        conf_pos.append(pos[i])

    for u, v in internal_bonds:
        if mol.GetBondBetweenAtoms(u, v) is None:
            mol.AddBond(u, v, Chem.BondType.SINGLE)

    lat = structure_lattice(struct)
    for u, v, eff in periodic_bonds:
        img_v = pos[v] + eff @ lat        # where v's bonded image sits
        img_u = pos[u] - eff @ lat
        for host, target in ((u, img_v), (v, img_u)):
            sym = struct[host].specie.symbol
            if cap == "H":
                cap_atom = Chem.Atom("H")
                blen = H_CAP_LEN.get(sym, 1.05)
            else:
                cap_atom = Chem.Atom(0)   # [*] wildcard
                blen = 1.5
            vec = target - pos[host]
            nrm = np.linalg.norm(vec)
            cpos = pos[host] + vec / max(nrm, 1e-6) * blen
            idx = mol.AddAtom(cap_atom)
            mol.AddBond(host, idx, Chem.BondType.SINGLE)
            conf_pos.append(cpos)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(conf_pos):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mol.AddConformer(conf)
    return mol


def structure_lattice(struct):
    return struct.lattice.matrix


import signal as _signal


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, seconds):
    """Abort fn() with _Timeout if it exceeds `seconds` (POSIX only)."""
    if not hasattr(_signal, "SIGALRM"):
        return fn()
    def _h(sig, frm):
        raise _Timeout()
    old = _signal.signal(_signal.SIGALRM, _h)
    _signal.setitimer(_signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old)


PERCEPTION_MAX_ATOMS = 200   # cap for attempting whole-cell perception
PERCEPTION_TIMEOUT = 20.0    # seconds; enforced via killable subprocess


def _determine_bonds_subprocess(molblock, timeout):
    """
    Run rdDetermineBonds in a separate process so a C++ hang can be killed
    (signals can't interrupt the in-process C++ call). Returns SMILES or None.
    """
    import subprocess
    code = (
        "import sys;from rdkit import Chem;"
        "from rdkit.Chem import rdDetermineBonds;"
        "m=Chem.MolFromMolBlock(sys.stdin.read(),sanitize=False,removeHs=False);"
        "rdDetermineBonds.DetermineBonds(m,charge=0);"
        "print(Chem.MolToSmiles(m))")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           input=molblock, capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout.strip()
        return out if out else None
    except (subprocess.TimeoutExpired, Exception):
        return None


def perceive_and_smiles(mol: Chem.RWMol, has_star: bool):
    """
    Full bond-order perception for H-capped cells (star mode keeps single
    bonds). DetermineBonds is exponential worst-case on big conjugated
    cells and is C++ (uninterruptible in-process), so it runs in a
    subprocess with a hard timeout; on failure/timeout we fall back to the
    loosely-sanitized single-bond SMILES. Returns canonical SMILES or None.
    """
    m = mol.GetMol()
    if not has_star and m.GetNumHeavyAtoms() <= PERCEPTION_MAX_ATOMS:
        try:
            mb = Chem.MolToMolBlock(m)
            out = _determine_bonds_subprocess(mb, PERCEPTION_TIMEOUT)
            if out:
                return out
        except Exception:
            pass
    # Fallback: sanitize loosely, keep single bonds
    try:
        m2 = Chem.Mol(m)
        Chem.SanitizeMol(
            m2,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Aux topology tokens
# ----------------------------------------------------------------------
def topology_tokens(struct: Structure):
    lengths = np.array(struct.lattice.abc)
    axis = int(np.argmin(lengths))
    others = [i for i in range(3) if i != axis]
    layered = lengths[axis] < 0.55 * min(lengths[o] for o in others) \
        and lengths[axis] < 15.0
    return {
        "n_atoms": len(struct),
        "density_g_cm3": round(float(struct.density), 3),
        "volume_A3": round(struct.volume, 1),
        "stacking_len_A": round(float(lengths[axis]), 2) if layered else "",
        "layered": "yes" if layered else "no",
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def process_cif(path: Path, cap: str, max_atoms: int):
    row = {"file": path.name}
    try:
        struct = Structure.from_file(str(path))
    except Exception as e:
        row["status"] = f"READ_FAILED: {e}"
        return row
    if len(struct) > max_atoms:
        row["status"] = f"SKIPPED_{len(struct)}_atoms"
        return row

    row.update(topology_tokens(struct))
    try:
        G = build_quotient_graph(struct)
        G = prune_overbonded(struct, G)
        pos, internal, periodic = unwrap(struct, G)
        row["n_pbc_bonds_cut"] = len(periodic)

        # --- full unit-cell SMILES ------------------------------------
        mol = to_rdkit(struct, pos, internal, periodic, cap=cap)
        smi = perceive_and_smiles(
            mol, has_star=(cap == "star" and len(periodic) > 0))
        row["unit_smiles"] = smi or ""

        # --- unique fragments -----------------------------------------
        frags = set()
        if smi:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is not None:
                for f in Chem.GetMolFrags(m, asMols=True,
                                          sanitizeFrags=False):
                    fs = Chem.MolToSmiles(f)
                    if fs:
                        frags.add(fs)
        row["n_unique_fragments"] = len(frags)
        row["frag_smiles"] = ".".join(sorted(frags, key=len, reverse=True))
        row["status"] = "OK" if smi else "SMILES_FAILED"
    except Exception as e:
        row["status"] = f"ERROR: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CIF file or directory of CIFs")
    ap.add_argument("--out", default="cof_smiles.csv")
    ap.add_argument("--cap", choices=["star", "H"], default="star",
                    help="cap cut periodic bonds with [*] (default) or H")
    ap.add_argument("--max-atoms", type=int, default=2000)
    args = ap.parse_args()

    p = Path(args.input)
    cifs = [p] if p.is_file() else sorted(p.glob("*.cif"))
    if not cifs:
        sys.exit("No CIFs found.")

    rows = []
    total = len(cifs)
    for k, c in enumerate(cifs, 1):
        sys.stdout.write(f"\r[{k}/{total}] processing {c.name:40s}")
        sys.stdout.flush()
        row = process_cif(c, args.cap, args.max_atoms)
        rows.append(row)
        smi = row.get("unit_smiles", "")
        sys.stdout.write(
            f"\r[{k}/{total}] {row['file']:36s} {row.get('status',''):14s} "
            f"{smi[:44] + ('...' if len(smi) > 44 else ''):47s}\n")
        sys.stdout.flush()

    cols = ["file", "status", "n_atoms", "density_g_cm3", "volume_A3",
            "layered", "stacking_len_A", "n_pbc_bonds_cut",
            "n_unique_fragments", "unit_smiles", "frag_smiles"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
