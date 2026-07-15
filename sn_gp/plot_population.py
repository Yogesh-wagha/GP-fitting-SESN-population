"""
plot_population.py  --  figures from fit/population_measurements.csv.
Markers are keyed by SN type. One figure per --plot.

    python plot_population.py --plot durlum_r          # duration-lum, r band
    python plot_population.py --plot durlum_g           # duration-lum, g band
    python plot_population.py --plot durlum_brightest   # duration-lum, brightest band
    python plot_population.py --plot color_hist         # g-r peak colour histogram
    python plot_population.py --plot dtpeak_hist        # peak-time separation histogram
    python plot_population.py --plot all                # make all of the above

Duration axis = rest-frame FWHM [days]. Luminosity axis = peak ABSOLUTE mag
(brighter = up; y-axis inverted). Magnitudes are NOT K-corrected (noted on plot).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

CSV = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "population_plots")
os.makedirs(OUTDIR, exist_ok=True)



# TYPE_STYLE = {
#     "SN Ic-BL":  dict(marker="s", color="#8A0D67", label="Ic-BL"),
#     "SN Ic-BL?": dict(marker="s", color="#8A0D67", label="Ic-BL?"),
#     "SN Ic":     dict(marker="o", color="#1017F1", label="Ic"),
#     "SN Ib":     dict(marker="^", color="#1017F1", label="Ib"),
#     "SN Ib/c":   dict(marker="D", color="#1017F1", label="Ib/c"),
#     "SLSN-I":    dict(marker="*", color="#1FCFD8", label="SLSN-I"),
# }

TYPE_STYLE = {
    "SN Ic-BL":  dict(marker="s", color="#FF0000", label="Ic-BL"),
    "SN Ic-BL?": dict(marker="s", color="#FF0000", label="Ic-BL?"),
    "SN Ic":     dict(marker="o", color="#3300FF", label="Ic"),
    "SN Ib":     dict(marker="^", color="#FFA600", label="Ib"),
    "SN Ib/c":   dict(marker="D", color="#E100FF", label="Ib/c"),
    "SLSN-I":    dict(marker="*", color="#00FF44", label="SLSN-I"),
}

TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]

HIST_COLOR = {
    "SN Ic-BL":  "#FF0000", "SN Ic-BL?": "#FF0000",
    "SN Ic":     "#3300FF", "SN Ib":     "#FFA600", "SN Ib/c":   "#E100FF",
    "SLSN-I":    "#00FF44",
}

LINESTYLE = {
    "SN Ic": "-", "SN Ib": "--", "SN Ib/c": ":",
    "SN Ic-BL": "-", "SN Ic-BL?": "--", "SLSN-I": "-",
}

CONSTRAINING_DAYS = 16.0     # Perley: drop lower limits below this (uninformative)
FAST_RISE_DAYS   = 8.0       # ...unless rise measured and faster than this

def style_for(t):
    return TYPE_STYLE.get(str(t), dict(marker="x", color="grey", label=str(t)))


def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    return pd.read_csv(CSV)


def _scatter_by_type(ax, x, y, types):
    """Scatter with one legend entry per type present."""
    seen = set()
    for xi, yi, t in zip(x, y, types):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        s = style_for(t)
        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])
        ax.scatter(xi, yi, marker=s["marker"], color=s["color"], s=20,
                   edgecolor="k", linewidth=0.4, alpha=0.85, label=lbl)




def durlum(df, band, show_limits=False):
    if band == "brightest":
        xcol, ycol = "fwhm_brightest", "M_brightest"
        ll_col, lim_col = "fwhm_brightest_ll", "brightest_is_limit"
        title = "Duration-luminosity (brightest of g/r)"
    else:
        xcol, ycol = f"fwhm_{band}", f"M_{band}"
        ll_col, lim_col = f"fwhm_{band}_ll", f"{band}_is_limit"
        title = f"Duration-luminosity ({band} band)"

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # fully-measured -> filled markers
    meas = np.isfinite(df[xcol]) & np.isfinite(df[ycol]) & ~df[lim_col].fillna(False)
    _scatter_by_type(ax, df[xcol][meas], df[ycol][meas], df["type"][meas])

    if show_limits:
        lim = df[lim_col].fillna(False) & np.isfinite(df[ll_col]) & np.isfinite(df[ycol])
        constraining = df[ll_col] >= CONSTRAINING_DAYS
        if f"rise_{band}" in df.columns:
            fast = np.isfinite(df[f"rise_{band}"]) & (df[f"rise_{band}"] < FAST_RISE_DAYS)
            lim &= (constraining | fast)
        else:
            lim &= constraining
        n_lim = int(lim.sum())
        seen = set()
        for xi, yi, t in zip(df[ll_col][lim], df[ycol][lim], df["type"][lim]):
            s = style_for(t)
            lbl = f"{s['label']} (limit)" if f"{s['label']} (limit)" not in seen else None
            seen.add(f"{s['label']} (limit)")
            ax.scatter(xi, yi, marker=s["marker"], s=26,
                       facecolors="none", edgecolors=s["color"],
                       linewidth=1.1, alpha=0.9, label=lbl)
        print(f"   overlaid {n_lim} hollow lower-limit markers")

    ax.set_xlabel("rest-frame FWHM [days]")
    ax.set_ylabel("peak absolute magnitude")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.text(0.99, 0.01, "not K-corrected", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="grey")
    ax.legend(fontsize=8, title="type", loc="best")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"durlum_{band}{'_lim' if show_limits else ''}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}")

def hist_by_type_step(df, band, quantity, bins=20, density=False):
    """One step-histogram by type for a single durlum quantity.
    quantity: 'duration' (rest-frame FWHM) or 'luminosity' (peak abs mag).
    """
    if quantity == "duration":
        col = "fwhm_brightest" if band == "brightest" else f"fwhm_{band}"
        xlabel, invert = "rest-frame FWHM [days]", False
    elif quantity == "luminosity":
        col = "M_brightest" if band == "brightest" else f"M_{band}"
        xlabel, invert = "peak absolute magnitude", False
    else:
        raise ValueError("quantity must be 'duration' or 'luminosity'")

    lim_col = "brightest_is_limit" if band == "brightest" else f"{band}_is_limit"
    meas = np.isfinite(df[col]) & ~df[lim_col].fillna(False)  # drop lower limits
    v, t = df[col][meas], df["type"][meas]
    edges = np.linspace(np.nanmin(v), np.nanmax(v), bins + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for typ in TYPE_ORDER:
        m = (t == typ)
        if not m.any():
            continue
        ax.hist(v[m], bins=edges, histtype="step",
                color=HIST_COLOR[typ], linestyle=LINESTYLE[typ],
                linewidth=1.6, density=density, label=TYPE_STYLE[typ]["label"])

    if invert:
        ax.invert_xaxis()   # for luminosity: uncomment-equivalent -> brighter right
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "number of events")
    ax.set_title(f"{quantity.capitalize()} distribution ({band} band)")
    ax.legend(fontsize=8, title="type", loc="best")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"hist_{quantity}_{band}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}  ({int(meas.sum())} measured events)")

def _hist_by_type(df, col, xlabel, fname, bins=25):
    sub = df[np.isfinite(df[col])]
    if sub.empty:
        print(f"no finite {col}; skipping {fname}"); return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.histogram_bin_edges(sub[col].values, bins=bins)
    for t in TYPE_ORDER:
        d = sub.loc[sub["type"] == t, col].values
        if d.size == 0:
            continue
        ax.hist(d, bins=edges, histtype="step",
                color=HIST_COLOR[t], label=TYPE_STYLE[t]["label"], linewidth=2.2, linestyle=LINESTYLE[t])
    other = sub[~sub["type"].isin(HIST_COLOR)]
    if len(other):
        ax.hist(other[col].values, bins=edges, histtype="step",
                color="grey", label="other", linewidth=2.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("number of events")
    ax.legend(fontsize=8, title="type")
    ax.grid(alpha=0.2, axis="y")
    out = os.path.join(OUTDIR, fname)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} events)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", required=True,
        choices=["durlum_g", "durlum_r", "durlum_brightest",
                 "durhist_g", "durhist_r", "durhist_brightest",
                 "lumhist_g", "lumhist_r", "lumhist_brightest",
                 "color_hist", "dtpeak_hist", "all"])
    ap.add_argument("--color-mode", default="rpeak",
                    choices=["rpeak", "gpeak", "ownpeak"],
                    help="which colour definition to histogram")
    ap.add_argument("--limits", action="store_true",
        help="overlay constraining lower-limit events as arrows")
    a = ap.parse_args()
    df = _load()

    def do(p):
        if p == "durlum_g":                durlum(df, "g", show_limits=a.limits)
        elif p == "durlum_r":              durlum(df, "r", show_limits=a.limits)
        elif p == "durlum_brightest":      durlum(df, "brightest", show_limits=a.limits)
        elif p == "color_hist":
            colmap = {"rpeak": ("color_at_rpeak", "g - r at r-peak epoch"),
                      "gpeak": ("color_at_gpeak", "g - r at g-peak epoch"),
                      "ownpeak": ("color_gr", "g - r (own peaks)")}
            col, lab = colmap[a.color_mode]
            _hist_by_type(df, col, f"peak colour  {lab}  [mag]",
                          f"color_hist_{a.color_mode}.png")
        elif p == "dtpeak_hist":
            _hist_by_type(df, "dt_peak_g_minus_r",
                          "peak-time separation  t_g - t_r  [days]", "dtpeak_hist.png")
        elif p.startswith("durhist_"):
            hist_by_type_step(df, p.split("_")[1], "duration")
        elif p.startswith("lumhist_"):
            hist_by_type_step(df, p.split("_")[1], "luminosity")
        

    if a.plot == "all":
        for p in ("durlum_g", "durlum_r", "durlum_brightest",
                  "color_hist", "dtpeak_hist"):
            do(p)
    else:
        do(a.plot)


if __name__ == "__main__":
    main()
