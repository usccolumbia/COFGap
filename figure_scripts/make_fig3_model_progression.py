"""Figure 3: model progression (Table 3, 216 COFs, family-grouped 5-fold CV)."""
import matplotlib.pyplot as plt
rows = [("xTB1\nStruct.",0.274,0.577),("xTB1\nBasic",0.306,0.563),("xTB2\nRich",0.357,0.549),
        ("xTB2\n+Struct.",0.573,0.438),("xTB2\n+RDKit",0.575,0.432),("xTB3\nRaw Gap",0.591,0.433),
        ("xTB4\nBase",0.578,0.450),("xTB4\nPhysics",0.624,0.381),("xTB4\nEnsemble",0.640,0.380),
        ("xTB5\nv3",0.645,0.381),("xTB6\nv3.5",0.656,0.373)]
labels,r2,mae = zip(*rows); x = range(len(rows))
plt.rcParams.update({"font.family":"Arial","font.size":11})
fig, ax = plt.subplots(figsize=(11,4.6), dpi=300)
ax.bar(x, r2, width=0.55, color=["#3d6079"]*(len(rows)-1)+["#0d2643"], edgecolor="white", zorder=2)
ax.axhline(0.625, color="#7a7a7a", ls=":", lw=1.4, zorder=3)
ax.text(-0.45, 0.632, "DFT fragment + structure (RF), $R^2$ = 0.625", color="#5a5a5a", fontsize=9)
ax.set_ylim(0.2,0.72); ax.set_ylabel("$R^2$ Score", color="#1f3b52", fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
ax2 = ax.twinx()
ax2.plot(x, mae, color="#e63946", lw=2.5, marker="o", ms=7, mfc="white", mew=2, zorder=4)
ax2.set_ylim(0.30,0.65); ax2.set_ylabel("MAE (eV)", color="#e63946", fontweight="bold")
ax2.tick_params(axis="y", colors="#e63946")
for a in (ax, ax2):
    for s in a.spines.values(): s.set_color("#cccccc")
fig.tight_layout(); fig.savefig("figures/pvfig3.png")
