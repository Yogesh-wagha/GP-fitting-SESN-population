"""
plot_detections.py
Plot ONLY the detections of one SN light curve, phase relative to peak.
No isolated-point filtering here on purpose -- we want to SEE everything,
including stray points far from the peak, during visual review.

Used standalone (saves a PNG) and imported by review_app.py (the review site).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless: no GUI, safe inside a web server
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ---- paths ----
BASE        = "/home/yogesh1729/myWork/air_phd/gaussian_process_BTS"
CSV_DIR     = os.path.join(BASE, "BTS_csv")
BTS_CATALOG = "/home/yogesh1729/myWork/air_phd/BTS_all.csv"

G_FILTERS = ["ztfg", "sdssg"]
R_FILTERS = ["ztfr", "sdssr"]
MARKERS   = {"ztfg": "o", "sdssg": "s", "ztfr": "o", "sdssr": "s"}

# ---- cache the catalog so we read it once, not once per plot ----
_cat = None
def _catalog():
    global _cat
    if _cat is None:
        _cat = (pd.read_csv(BTS_CATALOG)
                  .drop_duplicates("ZTFID")
                  .set_index("ZTFID"))
    return _cat

def _lookup(target, col):
    cat = _catalog()
    if col in cat.columns and target in cat.index:
        return cat.loc[target, col]
    return None


def plot_detections(target, savepath=None):
    """Render the detection light curve. Returns the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(9, 4.2))

    df = pd.read_csv(os.path.join(CSV_DIR, f"{target}.csv"))
    df = df[df["origin"] != "alert_fp"]
    df = df[df["filter"].isin(G_FILTERS + R_FILTERS)].copy()

    # redshift -> absolute mag; fall back to apparent mag if no usable z
    z = pd.to_numeric(_lookup(target, "redshift"), errors="coerce")
    if pd.notna(z) and z > 0:
        dL_pc = cosmo.luminosity_distance(float(z)).value * 1e6
        df["y"] = df["mag"] - 5 * np.log10(dL_pc) + 5
        ylabel, ztxt = "Absolute magnitude", f"z = {float(z):.4f}"
    else:
        df["y"] = df["mag"]
        ylabel, ztxt = "Apparent magnitude", "no z (apparent)"

    cls = _lookup(target, "type")
    cls = "" if cls is None or (isinstance(cls, float) and pd.isna(cls)) else str(cls)

    g = df[df["filter"].isin(G_FILTERS)]
    r = df[df["filter"].isin(R_FILTERS)]
    bands = {b: d for b, d in (("g", g), ("r", r)) if not d.empty}

    if not bands:
        ax.text(0.5, 0.5, f"{target}: no g/r detections", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=110); plt.close(fig)
        return fig

    # reference peak = brightest (min mag) of the two bands
    ref_band = min(bands, key=lambda b: bands[b]["y"].min())
    peak_i   = bands[ref_band]["y"].idxmin()
    peak_mjd = df.loc[peak_i, "mjd"]
    peak_y   = df.loc[peak_i, "y"]
    df["phase"] = df["mjd"] - peak_mjd

    g = df[df["filter"].isin(G_FILTERS)]
    r = df[df["filter"].isin(R_FILTERS)]
    for sub, color in ((g, "green"), (r, "red")):
        for filt, fd in sub.groupby("filter"):
            ax.errorbar(fd["phase"], fd["y"], yerr=fd["magerr"],
                        fmt=MARKERS.get(filt, "o"), color=color, mfc="none",
                        ms=5, elinewidth=1, capsize=2, label=filt)

    ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Phase from peak [days]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{target}   {ztxt}   peak({ref_band})={peak_y:.2f}   {cls}")
    ax.legend(title="filter", fontsize=8)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=110)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else "ZTF19abafmwj"
    out = f"{tgt}.png"
    plot_detections(tgt, savepath=out)
    print("saved", out)
