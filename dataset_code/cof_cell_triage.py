#!/usr/bin/env python3
"""
cof_cell_triage.py
==================
Batch triage of COF CIFs for band-gap DFT calculations.

For each CIF in an input directory this script:
  1. Reads the structure (tolerant to messy experimental CIFs).
  2. Finds the true primitive cell (tries several symprec values; keeps the
     smallest valid reduction).
  3. Detects the stacking axis and counts distinct 2D layers per cell; checks
     whether the cell can be halved along the stacking axis (duplicate layers
     related by pure translation).
  4. Optionally builds a monolayer model (single layer + vacuum) for uniform
     2D-descriptor screening.
  5. Flags metal-containing frameworks (band gaps dominated by metal states,
     as excluded in the deCOFpose paper).
  6. Classifies DFT tractability by atom count and suggests a k-mesh
     (sparse in-plane, denser along stacking).
  7. Writes a summary CSV + reduced CIFs to an output directory.

Usage:
    python cof_cell_triage.py <cif_dir> [--out OUTDIR] [--monolayer]
                              [--vacuum 20.0] [--symprec 1e-3 1e-2 0.1]

Dependencies: pymatgen, spglib, numpy (pip install pymatgen)
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

# Elements that disqualify a "pure organic" COF (metals + metalloids commonly
# intercalated). B and Si are kept: boronate/boroxine and Si-COFs are standard.
ORGANIC_OK = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}

TRACTABILITY = [
    (150,  "easy",      "hybrid (HSE06/PBE0) fine, plane-wave OK"),
    (400,  "moderate",  "hybrid OK in NAO/Gaussian codes (FHI-aims, CRYSTAL, CP2K ADMM)"),
    (900,  "hard",      "GGA fine; hybrid only Gamma-point CP2K/ONETEP-class"),
    (10**9, "very_hard", "GGA/DFTB screening only; hybrid single-points impractical"),
]


def read_structure(path: Path):
    """Tolerant CIF reading: try occupancy/site merging tolerances."""
    for kwargs in ({}, {"occupancy_tolerance": 2.0},
                   {"occupancy_tolerance": 2.0, "merge_tol": 0.01}):
        try:
            return Structure.from_file(str(path), **kwargs)
        except Exception:
            continue
    return None


def best_primitive(struct: Structure, symprecs):
    """Try several symprec values; return smallest primitive cell found."""
    best = struct
    best_prec = None
    for sp in symprecs:
        try:
            sga = SpacegroupAnalyzer(struct, symprec=sp)
            prim = sga.get_primitive_standard_structure()
            # Sanity: composition must be commensurate
            if len(prim) < len(best) and _same_composition(struct, prim):
                best, best_prec = prim, sp
        except Exception:
            continue
    return best, best_prec


def _same_composition(a: Structure, b: Structure) -> bool:
    return a.composition.reduced_composition == b.composition.reduced_composition


def detect_stacking_axis(struct: Structure):
    """
    Heuristic: the stacking axis of a layered COF is the SHORTEST lattice
    vector (typically 3-8 A interlayer vs 20-50 A in-plane). Returns axis
    index or None if the cell doesn't look layered.
    """
    lengths = np.array(struct.lattice.abc)
    axis = int(np.argmin(lengths))
    others = [i for i in range(3) if i != axis]
    # Layered if stacking axis much shorter than in-plane vectors
    if lengths[axis] < 0.55 * min(lengths[o] for o in others) and lengths[axis] < 15.0:
        return axis
    return None


def count_layers(struct: Structure, axis: int):
    """
    Cluster atoms by fractional coordinate along the stacking axis to count
    layers. Uses a gap-based 1D clustering with periodic wrap.
    """
    fc = np.sort(struct.frac_coords[:, axis] % 1.0)
    if len(fc) < 2:
        return 1
    # Gap threshold: interlayer gap in fractional units; ~1.5 A real-space
    thresh = 1.5 / struct.lattice.abc[axis]
    gaps = np.diff(np.concatenate([fc, [fc[0] + 1.0]]))
    n_layers = int(np.sum(gaps > thresh))
    return max(n_layers, 1)


def try_halve_stacking(struct: Structure, axis: int):
    """
    If the cell contains 2 layers related by a pure translation of 0.5 along
    the stacking axis (possibly with in-plane shift), the cell can be halved.
    Test: does translating by t = (dx, dy, 0.5) map the structure onto itself
    for some in-plane (dx, dy) taken from actual atom pairs?
    Returns halved Structure or None.
    """
    n = len(struct)
    if n % 2:
        return None
    fc = struct.frac_coords % 1.0
    species = [s.species_string for s in struct]

    # Candidate translations: from the first atom to every same-species atom
    # displaced ~0.5 along the stacking axis.
    ref = 0
    cands = []
    for j in range(n):
        if j == ref or species[j] != species[ref]:
            continue
        t = (fc[j] - fc[ref]) % 1.0
        if abs(t[axis] - 0.5) < 0.02:
            cands.append(t)

    for t in cands[:40]:  # cap the search
        # Build candidate halved cell: replace stacking vector with the
        # cartesian translation t (valid for eclipsed AND inclined AA'
        # stacking, where t has an in-plane component).
        try:
            m = struct.lattice.matrix.copy()
            m[axis] = t @ m  # cartesian translation vector
            new_latt = Lattice(m)
            if new_latt.volume < 1.0:
                continue
            new_fc = [new_latt.get_fractional_coords(
                          struct.lattice.get_cartesian_coords(f)) % 1.0
                      for f in fc]
            half = Structure(new_latt, species, new_fc,
                             coords_are_cartesian=False)
            half.merge_sites(tol=0.3, mode="delete")
        except Exception:
            continue
        # Validate: exactly half the atoms, same composition, no collisions
        if len(half) != n // 2 or not _same_composition(struct, half):
            continue
        dmat = half.distance_matrix
        np.fill_diagonal(dmat, 1e9)
        if dmat.min() < 0.7:  # atoms crashed together -> bad translation
            continue
        return half
    return None


def make_monolayer(struct: Structure, axis: int, vacuum: float):
    """Extract one layer and add vacuum along the stacking axis."""
    fc = struct.frac_coords % 1.0
    coords = np.sort(fc[:, axis])
    thresh = 1.5 / struct.lattice.abc[axis]
    gaps = np.diff(np.concatenate([coords, [coords[0] + 1.0]]))
    if not np.any(gaps > thresh):
        return None
    # Rotate origin to the largest gap so one layer is contiguous
    gap_start = coords[int(np.argmax(gaps))] + gaps.max() / 2.0
    shifted = (fc[:, axis] - gap_start) % 1.0
    # First layer = first contiguous cluster
    order = np.argsort(shifted)
    sel = [order[0]]
    for k in order[1:]:
        if shifted[k] - shifted[sel[-1]] > thresh:
            break
        sel.append(k)
    sel = np.array(sel)

    latt = struct.lattice
    cart = struct.cart_coords[sel]
    species = [struct[i].species_string for i in sel]
    # New lattice: in-plane vectors kept, stacking axis -> orthogonal, len = span+vacuum
    m = latt.matrix.copy()
    normal = np.cross(m[(axis + 1) % 3], m[(axis + 2) % 3])
    normal /= np.linalg.norm(normal)
    span = cart @ normal
    thickness = span.max() - span.min()
    m[axis] = normal * (thickness + vacuum)
    mono = Structure(Lattice(m), species, cart, coords_are_cartesian=True)
    # center in vacuum
    mono.translate_sites(range(len(mono)),
                         normal * ((thickness + vacuum) / 2 - span.mean()),
                         frac_coords=False)
    return mono


def suggest_kmesh(struct: Structure, axis, kppra_stack=40.0):
    """Sparse in-plane (huge cell -> tiny BZ), denser along stacking."""
    mesh = [1, 1, 1]
    if axis is not None:
        c = struct.lattice.abc[axis]
        mesh[axis] = max(1, int(round(kppra_stack / c)))  # ~40/c_A points
    return "x".join(map(str, mesh))


def tractability(natoms):
    for lim, label, note in TRACTABILITY:
        if natoms <= lim:
            return label, note
    return "very_hard", ""


def process(path: Path, outdir: Path, symprecs, monolayer, vacuum):
    row = {"file": path.name}
    s = read_structure(path)
    if s is None:
        row["status"] = "READ_FAILED"
        return row
    row["n_atoms_input"] = len(s)
    elems = {el.symbol for el in s.composition.elements}
    row["has_metal"] = "yes" if (elems - ORGANIC_OK) else "no"
    row["metals"] = ",".join(sorted(elems - ORGANIC_OK))

    prim, prec = best_primitive(s, symprecs)
    row["n_atoms_primitive"] = len(prim)
    row["symprec_used"] = prec if prec is not None else ""
    try:
        row["spacegroup"] = SpacegroupAnalyzer(prim, symprec=0.1).get_space_group_symbol()
    except Exception:
        row["spacegroup"] = "?"

    axis = detect_stacking_axis(prim)
    row["layered"] = "yes" if axis is not None else "no"
    row["stacking_axis"] = "abc"[axis] if axis is not None else ""
    work = prim

    if axis is not None:
        row["layers_per_cell"] = count_layers(prim, axis)
        if row["layers_per_cell"] >= 2:
            half = try_halve_stacking(prim, axis)
            if half is not None:
                work = half
                row["halved_along_stack"] = "yes"
        row.setdefault("halved_along_stack", "no")
        d = work.lattice.abc[axis] / max(count_layers(work, axis), 1)
        row["interlayer_dist_A"] = round(d, 2)

    row["n_atoms_final"] = len(work)
    row["reduction_factor"] = round(len(s) / len(work), 2)
    label, note = tractability(len(work))
    row["dft_tractability"] = label
    row["dft_note"] = note
    row["suggested_kmesh"] = suggest_kmesh(work, axis)
    row["status"] = "OK"

    stem = path.stem
    work.to(filename=str(outdir / f"{stem}_reduced.cif"))
    if monolayer and axis is not None:
        mono = make_monolayer(work, axis, vacuum)
        if mono is not None:
            mono.to(filename=str(outdir / f"{stem}_monolayer.cif"))
            row["monolayer_atoms"] = len(mono)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("cif_dir")
    ap.add_argument("--out", default="cof_triage_out")
    ap.add_argument("--monolayer", action="store_true",
                    help="also write single-layer + vacuum models")
    ap.add_argument("--vacuum", type=float, default=20.0)
    ap.add_argument("--symprec", type=float, nargs="+",
                    default=[1e-3, 1e-2, 0.1])
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cifs = sorted(Path(args.cif_dir).glob("*.cif"))
    if not cifs:
        sys.exit(f"No CIFs found in {args.cif_dir}")

    rows = []
    for p in cifs:
        try:
            row = process(p, outdir, args.symprec, args.monolayer, args.vacuum)
        except Exception as e:
            row = {"file": p.name, "status": f"ERROR: {e}"}
        rows.append(row)
        print(f"{row.get('file'):40s} {row.get('status',''):10s} "
              f"{row.get('n_atoms_input','')}->{row.get('n_atoms_final','')} atoms  "
              f"[{row.get('dft_tractability','')}]")

    cols = ["file", "status", "n_atoms_input", "n_atoms_primitive",
            "halved_along_stack", "n_atoms_final", "reduction_factor",
            "monolayer_atoms", "layered", "stacking_axis", "layers_per_cell",
            "interlayer_dist_A", "spacegroup", "symprec_used", "has_metal",
            "metals", "dft_tractability", "suggested_kmesh", "dft_note"]
    with open(outdir / "triage_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary: {outdir/'triage_summary.csv'}")


if __name__ == "__main__":
    main()
