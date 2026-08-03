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
plt.rcParams.update({
    "font.size":        14,   # base size (default 10) — everything scales from this
    "axes.titlesize":   16,   # plot title
    "axes.labelsize":   15,   # x/y axis labels
    "xtick.labelsize":  13,   # tick numbers
    "ytick.labelsize":  13,
    "legend.fontsize":  16,   # legend text
    "legend.title_fontsize": 13,
})

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
    "SN Ic-BL?": dict(marker="s", color="#FF0000", label="Ic-BL"),
    "SN Ic":     dict(marker="o", color="#3300FF", label="Ic"),
    "SN Ib":     dict(marker="^", color="#FFA600", label="Ib"),
    "SN Ib/c":   dict(marker="D", color="#E100FF", label="Ib/c"),
    "SLSN-I":    dict(marker="*", color="#3DC566ED", label="SLSN-I"),
}

TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]

HIST_COLOR = {
    "SN Ic-BL":  "#FF0000", "SN Ic-BL?": "#FF0000",
    "SN Ic":     "#3300FF", "SN Ib":     "#FFA600", "SN Ib/c":   "#E100FF",
    "SLSN-I":    "#3DC566ED",
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


def durlum(df, band, show_limits=False, max_magerr=None):
    ycol, yerr_col = "M_rest_g", "M_rest_g_err"
    if band == "brightest":
        xcol = "fwhm_brightest"
        ll_col, lim_col = "fwhm_brightest_ll", "brightest_is_limit"
    else:
        xcol = f"fwhm_{band}"
        ll_col, lim_col = f"fwhm_{band}_ll", f"{band}_is_limit"
    title = "Duration-luminosity"

    fig, ax = plt.subplots(figsize=(7, 5))

    meas = np.isfinite(df[xcol]) & np.isfinite(df[ycol]) & ~df[lim_col].fillna(False)
    # error cap: drop events whose magnitude error exceeds the cap
    if max_magerr is not None and yerr_col in df.columns:
        too_big = df[yerr_col] > max_magerr
        n_drop = int((meas & too_big).sum())
        meas &= ~too_big
        print(f"   error cap {max_magerr} mag: dropped {n_drop} events with larger M error")

    seen = set()
    for _, r in df[meas].iterrows():
        s = style_for(r["type"])
        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])
        yerr = r[yerr_col] if (yerr_col in df.columns and np.isfinite(r[yerr_col])) else None
        ax.errorbar(r[xcol], r[ycol], yerr=yerr, fmt=s["marker"], color=s["color"],
                    ms=5, mec="k", mew=0.4, alpha=0.85, elinewidth=0.7,
                    capsize=1.5, label=lbl)

    if show_limits:
        lim = df[lim_col].fillna(False) & np.isfinite(df[ll_col]) & np.isfinite(df[ycol])
        constraining = df[ll_col] >= CONSTRAINING_DAYS
        if f"rise_{band}" in df.columns:
            fast = np.isfinite(df[f"rise_{band}"]) & (df[f"rise_{band}"] < FAST_RISE_DAYS)
            lim &= (constraining | fast)
        else:
            lim &= constraining
        if max_magerr is not None and yerr_col in df.columns:   # same cap on limits
            lim &= ~(df[yerr_col] > max_magerr)
        for _, r in df[lim].iterrows():
            s = style_for(r["type"])
            yerr = r[yerr_col] if (yerr_col in df.columns and np.isfinite(r[yerr_col])) else None
            ax.errorbar(r[ll_col], r[ycol], yerr=yerr, fmt=s["marker"],
                        mfc="none", mec=s["color"], ms=6, mew=1.1,
                        alpha=0.9, elinewidth=0.6, capsize=1.5)
        print(f"   overlaid {int(lim.sum())} hollow lower-limit markers")
    ax.axhline(-17.8, color="k", ls="--", lw=1.2, alpha=0.7)
    ax.text(125, -17.8, " -17.8", va="bottom", ha="left",
            fontsize=7, color="k", transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"rest-frame FWHM [days] ({band})")
    ax.set_ylabel("rest-frame g peak absolute magnitude")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(fontsize=10, title="type", loc="best")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"durlum_{band}{'_lim' if show_limits else ''}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}")

def rise_duration(df, band):
    """Rise time vs total duration (FWHM), rest-frame, coloured by type."""
    xcol = f"rise_{band}"
    ycol = "fwhm_brightest" if band == "brightest" else f"fwhm_{band}"
    if xcol not in df.columns:
        # rise measured per-band; fall back to r if brightest has no rise column
        xcol = "rise_r"; band = "r"; ycol = "fwhm_r"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    meas = np.isfinite(df[xcol]) & np.isfinite(df[ycol])
    _scatter_by_type(ax, df[xcol][meas], df[ycol][meas], df["type"][meas])
    # optional 1:1-ish guide: duration >= rise always, since duration = rise + fade
    lo = 0
    hi = float(np.nanmax(df[ycol][meas])) if meas.any() else 1
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8, alpha=0.5, label="duration = rise")
    ax.set_xlabel(f"rest-frame rise time [days] ({band})")
    ax.set_ylabel(f"rest-frame total duration FWHM [days] ({band})")
    ax.set_title(f"Rise time vs total duration ({band} band)")
    ax.legend(fontsize=8, title="type", loc="best")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"rise_duration_{band}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({int(meas.sum())} events)")

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
    seen_labels = set()
    for t in TYPE_ORDER:
        d = sub.loc[sub["type"] == t, col].values
        if d.size == 0:
            continue
        lab = TYPE_STYLE[t]["label"]
        lbl = lab if lab not in seen_labels else None
        seen_labels.add(lab)
        ax.hist(d, bins=edges, histtype="step",
                color=HIST_COLOR[t], label=lbl,        # None for the duplicate
                linewidth=2.2, linestyle=LINESTYLE[t])
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
                 "color_hist", "dtpeak_hist", "rise_duration_g", 
                 "rise_duration_r", "rise_duration_brightest", "all"])
    ap.add_argument("--color-mode", default="rpeak",
                    choices=["rpeak", "gpeak", "ownpeak", "10d"],
                    help="which colour definition for color_hist")
    ap.add_argument("--max-magerr", type=float, default=None,
        help="drop scatter points whose M_rest_g error exceeds this (mag); scatter only")
    ap.add_argument("--limits", action="store_true",
        help="overlay constraining lower-limit events as arrows")
    a = ap.parse_args()
    df = _load()

    def do(p):
        if p == "durlum_g":           durlum(df, "g", show_limits=a.limits, max_magerr=a.max_magerr)
        elif p == "durlum_r":         durlum(df, "r", show_limits=a.limits, max_magerr=a.max_magerr)
        elif p == "durlum_brightest": durlum(df, "brightest", show_limits=a.limits, max_magerr=a.max_magerr)
        elif p == "color_hist":
            cmap = {"rpeak": ("color_at_rpeak", "g-r at r-peak"),
                    "gpeak": ("color_at_gpeak", "g-r at g-peak"),
                    "ownpeak": ("color_gr", "g-r own-peak"),
                    "10d": ("color_10d", "g-r at +10d")}
            col, lab = cmap[a.color_mode]
            _hist_by_type(df, col, f"peak colour  {lab}  [mag]",
                          f"color_hist_{a.color_mode}.png")
        elif p == "dtpeak_hist":
            _hist_by_type(df, "dt_peak_g_minus_r",
                          "peak-time separation  t_g - t_r  [days]", "dtpeak_hist.png")
        elif p.startswith("durhist_"):
            hist_by_type_step(df, p.split("_")[1], "duration")
        elif p.startswith("lumhist_"):
            hist_by_type_step(df, p.split("_")[1], "luminosity")
        elif p.startswith("rise_duration_"):
            rise_duration(df, p.split("_")[-1])
        

    if a.plot == "all":
        for p in ("durlum_g", "durlum_r", "durlum_brightest",
                  "color_hist", "dtpeak_hist"):
            do(p)
    else:
        do(a.plot)


if __name__ == "__main__":
    main()
