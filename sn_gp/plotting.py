"""
plotting.py  --  one stacked panel per band (blue at top -> red at bottom),
sharing the time axis, each showing data + GP slice + 1/2 sigma. The AIC/BIC
box goes on the top panel. Many filters stay readable because they don't
overlap -- one row each.

predict_slice(name, wave_um) must return (phase_grid, mu, sd) in DATA units.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- MNRAS-style ----
try:
    plt.style.use("science")
except Exception:
    pass
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "legend.frameon": True, "legend.edgecolor": "k",
    "legend.facecolor": "white", "legend.framealpha": 1,
})


def plot_fit(obj, predict_slice, metrics, kernel_name, mean_name, gri=False):
    """obj: dict from config.load_object; predict_slice: callable;
       metrics: dict from metrics.compute_aic_bic.
       gri=True -> fixed 3x2 grid of the six g/r/i bands (those present).
       gri=False -> dynamic grid sized to however many bands the object has."""
    import math
    from config import GRI_FILTERS

    df = obj["df"]

    # choose which bands to show
    if gri:
        bands = [b for b in GRI_FILTERS if b in obj["bands"]]   # keep g/r/i order
        nrows, ncols = 3, 2
    else:
        bands = obj["bands"]                                    # blue -> red
        n = len(bands)
        ncols = 1 if n == 1 else (2 if n <= 6 else 3)           # near-square
        nrows = math.ceil(n / ncols)

    if not bands:
        raise ValueError(f"{obj['name']}: no bands to plot")

    # width scales with columns; height with rows (single-column-ish panels)
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(3.3 * ncols, 1.8 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(bands)))

    for ax, band, col in zip(flat, bands, cmap):
        sub = df[df["filter"] == band]
        wave = sub["wave_um"].iloc[0]
        tg, mu, sd = predict_slice(band, wave)
        ax.fill_between(tg, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.15)
        ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.30)
        ax.plot(tg, mu, color=col, lw=1.0)
        ax.errorbar(sub["phase"], sub["flux"], yerr=sub["fluxerr"],
                    fmt="o", ms=2.5, color="k", elinewidth=0.6, capsize=1.2)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.text(0.97, 0.90, band, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec=col, lw=0.8))

    # hide any unused panels in the grid
    for ax in flat[len(bands):]:
        ax.set_visible(False)

    # shared axis labels (one per side, not per panel)
    fig.supxlabel("phase from peak [days]", fontsize=9)
    fig.supylabel("flux [mJy]", fontsize=9)

    # AIC/BIC box OUTSIDE the axes, to the right
    txt = (f"{kernel_name} × wave\nmean: {mean_name}\n"
           f"k = {metrics['k']}\nlnL = {metrics['lnL']:.1f}\n"
           f"AIC = {metrics['AIC']:.1f}\nBIC = {metrics['BIC']:.1f}")
    fig.text(0.99, 0.5, txt, ha="right", va="center", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", lw=0.8))

    fig.suptitle(obj["name"], fontsize=9)
    # leave room on the right for the box
    fig.tight_layout(rect=[0, 0, 0.82, 0.96])

    # ---- to save instead of (or as well as) showing, uncomment: ----
    # fig.savefig(f"{obj['name']}_{kernel_name}_{mean_name}.pdf", bbox_inches="tight")

    plt.show()
    return fig