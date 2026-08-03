"""
plot_by_type.py  --  histogram and/or CDF of any measured quantity, for a
chosen SN type (or several types), using argparse. Generalises to any class.

    # peak abs mag of just Ic
    python plot_by_type.py --quantity M_rest_g --type "SN Ic" --kind hist
    python plot_by_type.py --quantity M_rest_g --type "SN Ic" --kind cdf

    # compare Ic and Ib on one figure
    python plot_by_type.py --quantity M_rest_g --type "SN Ic" "SN Ib" --kind both

    # any quantity works: fwhm_r, color_10d, dt_peak_g_minus_r, M_r, ...
    python plot_by_type.py --quantity fwhm_r --type "SN Ic-BL" --kind cdf

    # all six types, each its own curve, on one figure
    python plot_by_type.py --quantity M_rest_g --type all --kind cdf

Types accept the full BTS label ("SN Ic", "SN Ib", "SN Ic-BL", ...) or 'all'.
Multiple --type values overlay on the same figure, coloured by scheme.
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

# human-readable axis labels + whether brighter-left (invert x)
QUANTITY_INFO = {
    "M_rest_g":          ("rest-frame g peak abs mag (K-corrected)", True),
    "M_r":               ("r-band peak abs mag (obs-frame)", True),
    "M_g":               ("g-band peak abs mag (obs-frame)", True),
    "fwhm_r":            ("rest-frame FWHM r [days]", False),
    "fwhm_g":            ("rest-frame FWHM g [days]", False),
    "fwhm_brightest":    ("rest-frame FWHM brightest [days]", False),
    "color_gr":          ("g-r own-peak [mag]", False),
    "color_at_rpeak":    ("g-r at r-peak [mag]", False),
    "color_at_gpeak":    ("g-r at g-peak [mag]", False),
    "color_10d":         ("g-r at +10d [mag]", False),
    "dt_peak_g_minus_r": ("t_g - t_r [days]", False),
}


def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    return pd.read_csv(CSV)


def _color(t):  return TYPE_COLOR.get(t, "grey")
def _label(t):  return TYPE_LABEL.get(t, t)


def _resolve_types(types):
    if len(types) == 1 and types[0].lower() == "all":
        return TYPE_ORDER
    return types


def _tag(types):
    return "_".join(_label(t).replace("/", "") for t in types)


def hist(df, col, types, xlabel, invert_x, bins):
    fig, ax = plt.subplots(figsize=(7, 5))
    edges = None
    # shared bin edges across the chosen types
    allv = pd.to_numeric(df.loc[df["type"].isin(types), col],
                         errors="coerce").dropna().values
    if allv.size == 0:
        print(f"no finite {col} for {types}; skipping hist"); return
    edges = np.histogram_bin_edges(allv, bins=bins)
    for t in types:
        v = pd.to_numeric(df.loc[df["type"] == t, col], errors="coerce").dropna().values
        if v.size == 0:
            continue
        ax.hist(v, bins=edges, histtype="step", color=_color(t),
                linewidth=2.2, label=f"{_label(t)} (n={v.size})")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("number of events")
    ax.set_title(f"{xlabel}  --  histogram")
    ax.legend(fontsize=8, title="type"); ax.grid(alpha=0.2, axis="y")
    out = os.path.join(OUTDIR, f"hist_{col}_{_tag(types)}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}")


def cdf(df, col, types, xlabel, invert_x):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    any_drawn = False
    for t in types:
        v = pd.to_numeric(df.loc[df["type"] == t, col], errors="coerce").dropna().values
        if v.size < 2:
            continue
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=_color(t), lw=2.0,
                label=f"{_label(t)} (n={v.size})")
        any_drawn = True
    if not any_drawn:
        print(f"no type with >=2 finite {col}; skipping cdf"); plt.close(fig); return
    ax.set_ylim(0, 1.02)
    if invert_x:
        ax.axvline(-17.8, color="k", ls="--", lw=1.2, alpha=0.7)
        ax.text(-17.8, 0.6, " -17.8", rotation=90, va="bottom", ha="right",
                fontsize=10, color="k")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("cumulative fraction")
    ax.set_title(f"{xlabel}  --  CDF")
    ax.legend(fontsize=8, title="type"); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"cdf_{col}_{_tag(types)}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantity", required=True,
                    help="column to plot, e.g. M_rest_g, fwhm_r, color_10d")
    ap.add_argument("--type", nargs="+", required=True,
                    help='one or more SN types (e.g. "SN Ic" "SN Ib") or "all"')
    ap.add_argument("--kind", default="both", choices=["hist", "cdf", "both"])
    ap.add_argument("--bins", type=int, default=20)
    a = ap.parse_args()

    df = _load()
    col = a.quantity
    if col not in df.columns:
        raise SystemExit(f"column '{col}' not in {CSV}. Available: "
                         f"{[c for c in df.columns if c in QUANTITY_INFO]}")
    xlabel, invert_x = QUANTITY_INFO.get(col, (col, False))
    types = _resolve_types(a.type)

    # warn on unknown types (typo guard)
    known = set(df["type"].unique())
    for t in types:
        if t not in known:
            print(f"  warning: type '{t}' not present in data "
                  f"(available: {sorted(known)})")

    if a.kind in ("hist", "both"):
        hist(df, col, types, xlabel, invert_x, a.bins)
    if a.kind in ("cdf", "both"):
        cdf(df, col, types, xlabel, invert_x)


if __name__ == "__main__":
    main()