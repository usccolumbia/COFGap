"""Create the Figure 2-style DFT band-gap distribution for the xTB4 cohort."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# Match the visual styling used for the existing N = 315 figures.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.2,
        "grid.color": "#EBEBEB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.autolayout": True,
    }
)

PRIMARY_BLUE = "#1B4965"
ACCENT_RED = "#E63946"

HERE = Path(__file__).resolve().parent
FEATURE_TABLE = HERE.parent / "01_feature_table" / "xtb4_features.csv"
OUTPUT = HERE / "fig2_bandgap_distribution_N216.png"


def main() -> None:
    data = pd.read_csv(FEATURE_TABLE)
    bandgaps = pd.to_numeric(data["band_gap_eV"], errors="coerce").dropna()

    if len(bandgaps) != 216:
        raise ValueError(f"Expected 216 complete-xTB COFs, found {len(bandgaps)}.")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    ax.hist(
        bandgaps,
        bins=25,
        density=True,
        alpha=0.45,
        color=PRIMARY_BLUE,
        edgecolor="none",
    )
    x_grid = np.linspace(bandgaps.min() - 0.1, bandgaps.max() + 0.1, 512)
    ax.plot(x_grid, gaussian_kde(bandgaps)(x_grid), color=PRIMARY_BLUE, linewidth=2.5)
    ax.axvline(
        bandgaps.mean(),
        color=ACCENT_RED,
        linestyle=":",
        linewidth=2,
        label=f"Mean = {bandgaps.mean():.2f} eV",
    )
    ax.set_title("Distribution of Reference DFT Band Gaps ($N = 216$)", pad=10)
    ax.set_xlabel("DFT Band Gap (eV)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CCCCCC")

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT}")
    print(
        f"N={len(bandgaps)}, mean={bandgaps.mean():.3f} eV, "
        f"median={bandgaps.median():.3f} eV, range={bandgaps.min():.3f}-{bandgaps.max():.3f} eV"
    )


if __name__ == "__main__":
    main()
