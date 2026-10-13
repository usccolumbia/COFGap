"""Figure 4: pooled out-of-fold parity plot, TabPFN v3.5 (216 COFs, family-grouped 5-fold)."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
rows = list(csv.DictReader(open("DFT_replaced_xTB10_TabPFN_V3_5/oof_predictions.csv")))
y = np.array([float(r["band_gap_eV"]) for r in rows]); p = np.array([float(r["tabpfn_v3_5"]) for r in rows])
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=200)
ax.scatter(y, p, s=55, color="#0d2643", alpha=0.7, edgecolor="#0d2643", label=f"Test COFs ($N={len(y)}$)", zorder=3)
ax.plot([0, 5], [0, 5], "--", color="#666666", lw=1.5, label="Ideal 1:1 Parity")
ax.text(0.2, 4.75, f"$R^2 = {r2_score(y, p):.3f}$\nMAE$ = {mean_absolute_error(y, p):.3f}$ eV", va="top",
        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="#eef3f8", ec="#c8d4e0"))
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")
ax.set_xlabel("DFT Reference Band Gap (eV)"); ax.set_ylabel("Predicted Band Gap (eV)")
ax.grid(ls="--", alpha=0.35); ax.legend(loc="lower right")
for s in ax.spines.values(): s.set_color("#cccccc")
fig.tight_layout(); fig.savefig("figures/pvfig4.png")
