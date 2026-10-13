"""Figure 7: (a) linear fit of DFT band gap on PBE0 node+linker fragment gap (Ernst et al. Zenodo 14855145, 315 COFs);
(b) TabPFN v3.5 pooled out-of-fold predictions (216 COFs, family-grouped 5-fold)."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
D = ""
def num(v):
    try: return float(v.replace(",", "."))
    except (AttributeError, ValueError): return np.nan
z = list(csv.DictReader(open(D + "DFT_replaced_xTB3/00_Zenodo_14855145/COFsalldata_forplots.csv", encoding="utf-8-sig"), delimiter=";"))
xa = np.array([num(r["Node+Linker_HL_gap(eV)"]) for r in z]); ya = np.array([num(r["band gap (eV)"]) for r in z])
ok = np.isfinite(xa) & np.isfinite(ya); xa, ya = xa[ok], ya[ok]
pa = np.polyval(np.polyfit(xa, ya, 1), xa)
o = list(csv.DictReader(open(D + "DFT_replaced_xTB10_TabPFN_V3_5/oof_predictions.csv")))
yb = np.array([float(r["band_gap_eV"]) for r in o]); pb = np.array([float(r["tabpfn_v3_5"]) for r in o])
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), dpi=200, sharey=True)
for ax, y, p, c, tag in ((axes[0], ya, pa, "#d94a56", "(a)"), (axes[1], yb, pb, "#0d2643", "(b)")):
    ax.scatter(y, p, s=50, color=c, alpha=0.6, edgecolor=c, zorder=3)
    ax.plot([0, 5], [0, 5], "--", color="#888888", lw=1.6)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")
    ax.set_xlabel("DFT Reference Gap (eV)"); ax.grid(ls="--", alpha=0.35)
    ax.text(0.15, 4.8, tag, fontweight="bold", fontsize=14, va="top")
    ax.text(0.15, 4.25, f"$N = {len(y)}$\n$R^2 = {r2_score(y, p):.3f}$\nMAE$ = {mean_absolute_error(y, p):.3f}$ eV",
            va="top", fontsize=10.5, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    for s in ax.spines.values(): s.set_color("#cccccc")
axes[0].set_ylabel("Predicted Gap (eV)")
fig.tight_layout(); fig.savefig("figures/pvfig7.png")
print(r2_score(ya, pa), mean_absolute_error(ya, pa), r2_score(yb, pb))
