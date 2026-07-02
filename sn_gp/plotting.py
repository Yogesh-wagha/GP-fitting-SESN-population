"""
plotting.py  --  one stacked panel per band (blue at top -> red at bottom),
sharing the time axis, each showing data + GP slice + 1/2 sigma. The AIC/BIC
box goes on the top panel. Many filters stay readable because they don't
overlap -- one row each.

predict_slice(name, wave_um) must return (phase_grid, mu, sd) in DATA units.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
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


# band -> colour (consistent across panels: "g-like" is always green, etc.)
BAND_COLORS = {
    "g": "green", "r": "red", "i": "saddlebrown", "z": "black",
    "u": "blue",  "o": "orange", "c": "cyan",
    "b": "darkblue", "v": "olive",
    "uvw1": "violet", "uvw2": "darkviolet", "uvm2": "indigo",
}

def _family(filt):
    """Instrument family from the filter name prefix."""
    if filt.startswith("sdss"):  return "SDSS"
    if filt.startswith("ztf"):   return "ZTF"
    if filt.startswith("atlas"): return "ATLAS"
    if filt.startswith("uvot"):  return "UVOT"
    return "OTHER"

def _band_letter(filt):
    """The photometric band label used for colour (e.g. ztfg -> g, uvot::uvw1 -> uvw1)."""
    name = filt.split("::")[-1]            # uvot::uvw1 -> uvw1
    for pre in ("sdss", "ztf", "atlas"):
        if name.startswith(pre):
            name = name[len(pre):]
    return name                            # 'g','r','i','uvw1',...


def plot_fit(obj, predict_slice, metrics, kernel_name, mean_name,
             gri=False, peak_phase=0.0, t0_phase=None, outdir="figs"):
    import math
    df = obj["df"]
    present = obj["bands"]                  # blue -> red

    # ---- decide panels and which filters go in each ----
    if gri:
        gri_keep = {"g", "r", "i"}
        panels = ["SDSS", "ZTF"]            # left, right
        panel_filters = {
            "SDSS": [f for f in present if _family(f) == "SDSS" and _band_letter(f) in gri_keep],
            "ZTF":  [f for f in present if _family(f) == "ZTF"  and _band_letter(f) in gri_keep],
        }
        panels = [p for p in panels if panel_filters[p]]    # drop empty side
        nrows, ncols = 1, max(1, len(panels))
    else:
        fam_order = ["UVOT", "SDSS", "ZTF", "ATLAS", "OTHER"]
        fams = [f for f in fam_order if any(_family(b) == f for b in present)]
        panels = fams
        panel_filters = {fam: [b for b in present if _family(b) == fam] for fam in fams}
        n = len(panels)
        ncols = 1 if n == 1 else (2 if n <= 4 else 3)
        nrows = math.ceil(n / ncols)

    if not panels:
        raise ValueError(f"{obj['name']}: no bands to plot")

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=False,
                             figsize=(6 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, fam in zip(flat, panels):
        for band in panel_filters[fam]:
            sub = df[df["filter"] == band]
            wave = sub["wave_um"].iloc[0]
            col = BAND_COLORS.get(_band_letter(band), "grey")
            tg, mu, sd = predict_slice(band, wave)
            ax.fill_between(tg, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.12)
            ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.25)
            ax.plot(tg, mu, color=col, lw=1.0, label=band)
            ax.errorbar(sub["phase"], sub["flux"], yerr=sub["fluxerr"],
                        fmt="o", ms=2.5, color=col, mfc="none",
                        elinewidth=0.6, capsize=1.2)

        ax.axvline(peak_phase, color="grey", lw=0.8, ls="--")
        if t0_phase is not None:
            ax.axvline(t0_phase, color="purple", lw=0.8, ls=":")
        ax.set_ylim(bottom=0)
        ax.set_title(fam, fontsize=8)
        ax.legend(fontsize=7, loc="best")

    for ax in flat[len(panels):]:
        ax.set_visible(False)

    shown = [b for fl in panel_filters.values() for b in fl]
    used = df[df["filter"].isin(shown)]
    xpad = 0.05 * (used["phase"].max() - used["phase"].min() + 1e-9)
    flat[0].set_xlim(used["phase"].min() - xpad, used["phase"].max() + xpad)

    fig.supxlabel("phase from peak [days]", fontsize=9)
    fig.supylabel("flux [mJy]", fontsize=9)

    txt = (f"{kernel_name} × wave\nmean: {mean_name}\n"
           f"k = {metrics['k']}\nlnL = {metrics['lnL']:.1f}\n"
           f"AIC = {metrics['AIC']:.1f}\nBIC = {metrics['BIC']:.1f}")
    fig.text(0.99, 0.5, txt, ha="right", va="center", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", lw=0.8))

    fig.suptitle(obj["name"], fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 0.96])

    figdir = os.path.join(os.path.dirname(__file__), outdir)
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"{obj['name']}_{kernel_name}_{mean_name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    return fig