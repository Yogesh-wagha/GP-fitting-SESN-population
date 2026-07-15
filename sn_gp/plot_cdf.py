"""
plot_cdf.py  --  empirical cumulative distribution (CDF) plots by SN type,
for duration (rest-frame FWHM) and colour.

    python plot_cdf.py --plot duration            # rest-frame FWHM CDF, brightest band
    python plot_cdf.py --plot duration --band r    # r-band FWHM
    python plot_cdf.py --plot color                # colour CDF (own-peak by default)
    python plot_cdf.py --plot color --color-mode rpeak
    python plot_cdf.py --plot all
    python plot_cdf.py --plot duration --separate  # ALSO write one PNG per type

Overlaid CDF curves, one per type, coloured by your scheme. Vertical separation
between curves indicates distributional differences (what a KS test quantifies).

Duration CDF uses MEASURED FWHMs only (lower limits excluded); the count of
excluded limits is printed. For a censored-data CDF (Kaplan-Meier) see the note
in cdf_duration().
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

TYPE_COLOR = {
    "SN Ic-BL": "#FF0000", "SN Ic-BL?": "#FF0000",
    "SN Ic": "#3300FF", "SN Ib": "#FFA600", "SN Ib/c": "#E100FF",
    "SLSN-I": "#00FF44",
}
TYPE_LABEL = {
    "SN Ic-BL": "Ic-BL", "SN Ic-BL?": "Ic-BL?", "SN Ic": "Ic",
    "SN Ib": "Ib", "SN Ib/c": "Ib/c", "SLSN-I": "SLSN-I",
}
TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]


def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    return pd.read_csv(CSV)


def _ecdf(values):
    """Return (x_sorted, cdf) for a step ECDF."""
    v = np.sort(np.asarray(values, float))
    n = v.size
    y = np.arange(1, n + 1) / n
    return v, y


def _cdf_figure(df, col, xlabel, title, out, invert_x=False, separate=False):
    sub = df[np.isfinite(df[col])]
    if sub.empty:
        print(f"no finite {col}; skipping {title}"); return

    # overlaid figure
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for t in TYPE_ORDER:
        d = sub.loc[sub["type"] == t, col].values
        if d.size < 2:                              # need >=2 points for a meaningful CDF
            continue
        x, y = _ecdf(d)
        # draw as a step, extend to 0 and 1 at the ends
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=TYPE_COLOR[t], lw=2.0,
                label=f"{TYPE_LABEL[t]} (n={d.size})")
    ax.set_xlabel(xlabel); ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(title)
    ax.legend(fontsize=8, title="type", loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} events across types)")

    if separate:
        base, ext = os.path.splitext(out)
        for t in TYPE_ORDER:
            d = sub.loc[sub["type"] == t, col].values
            if d.size < 2:
                continue
            x, y = _ecdf(d)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                    drawstyle="steps-post", color=TYPE_COLOR[t], lw=2.2)
            ax.set_xlabel(xlabel); ax.set_ylabel("cumulative fraction")
            ax.set_ylim(0, 1.02)
            if invert_x:
                ax.invert_xaxis()
            ax.set_title(f"{title} -- {TYPE_LABEL[t]} (n={d.size})")
            ax.grid(alpha=0.2)
            outi = f"{base}_{TYPE_LABEL[t].replace('/', '')}{ext}"
            fig.tight_layout(); fig.savefig(outi, dpi=200); plt.close(fig)
            print(f"  wrote {outi}")


def cdf_duration(df, band, separate=False):
    col = "fwhm_brightest" if band == "brightest" else f"fwhm_{band}"
    lim_col = "brightest_is_limit" if band == "brightest" else f"{band}_is_limit"
    # measured only (exclude lower limits)
    n_lim = int((df[lim_col].fillna(False) &
                 ~np.isfinite(df[col])).sum()) if lim_col in df.columns else 0
    n_lim_total = int(df[lim_col].fillna(False).sum()) if lim_col in df.columns else 0
    print(f"duration CDF ({band}): excluding {n_lim_total} lower-limit events "
          f"(measured-only CDF; see KM note for censored version)")
    _cdf_figure(df, col,
                "rest-frame FWHM [days]",
                f"Duration CDF ({band} band, measured only)",
                os.path.join(OUTDIR, f"cdf_duration_{band}.png"),
                invert_x=False, separate=separate)
    # ---- Kaplan-Meier note ----
    # To include lower limits properly, fit a KM survival curve per type treating
    # is_limit events as right-censored at fwhm_ll, then plot 1 - S(t). Requires
    # e.g. lifelines.KaplanMeierFitter. Left as an extension if a censored CDF is
    # needed for a quantitative claim.


def cdf_color(df, color_mode, separate=False):
    colmap = {"rpeak": ("color_at_rpeak", "g - r at r-peak epoch"),
              "gpeak": ("color_at_gpeak", "g - r at g-peak epoch"),
              "ownpeak": ("color_gr", "g - r (own peaks)")}
    if color_mode not in colmap:
        color_mode = "ownpeak"
    col, lab = colmap[color_mode]
    if col not in df.columns:
        print(f"column {col} not in CSV (re-run measure_population with the new "
              f"colour columns); falling back to color_gr")
        col, lab = "color_gr", "g - r (own peaks)"
    _cdf_figure(df, col,
                f"colour  {lab}  [mag]",
                f"Colour CDF ({lab})",
                os.path.join(OUTDIR, f"cdf_color_{color_mode}.png"),
                invert_x=False, separate=separate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", required=True,
                    choices=["duration", "color", "all"])
    ap.add_argument("--band", default="brightest", choices=["g", "r", "brightest"])
    ap.add_argument("--color-mode", default="ownpeak",
                    choices=["rpeak", "gpeak", "ownpeak"])
    ap.add_argument("--separate", action="store_true",
                    help="ALSO write one CDF per type as separate PNGs")
    a = ap.parse_args()
    df = _load()

    if a.plot in ("duration", "all"):
        cdf_duration(df, a.band, separate=a.separate)
    if a.plot in ("color", "all"):
        cdf_color(df, a.color_mode, separate=a.separate)


if __name__ == "__main__":
    main()