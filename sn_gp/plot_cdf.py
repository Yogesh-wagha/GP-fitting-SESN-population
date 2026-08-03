"""
plot_cdf.py  --  empirical CDF plots by SN type for duration, colour, luminosity.
Plain step CDFs, one curve per type, coloured by scheme.

    python plot_cdf.py --plot duration
    python plot_cdf.py --plot duration --band r
    python plot_cdf.py --plot color --color-mode 10d     # g-r at +10d
    python plot_cdf.py --plot color --color-mode rpeak
    python plot_cdf.py --plot luminosity                  # M_rest_g CDF
    python plot_cdf.py --plot all

Duration = rest-frame FWHM. Luminosity = K-corrected rest-frame g abs mag.
Colours are observed-frame. Duration CDF uses measured FWHMs only (lower
limits excluded).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
plt.rcParams.update({
    "font.size":        14,   # base size (default 10) — everything scales from this
    "axes.titlesize":   16,   # plot title
    "axes.labelsize":   15,   # x/y axis labels
    "xtick.labelsize":  13,   # tick numbers
    "ytick.labelsize":  13,
    "legend.fontsize":  12,   # legend text
    "legend.title_fontsize": 13,
})
import config

CSV = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "population_plots")
os.makedirs(OUTDIR, exist_ok=True)

TYPE_COLOR = {
    "SN Ic-BL": "#FF0000", "SN Ic-BL?": "#FF0000",
    "SN Ic": "#3300FF", "SN Ib": "#FFA600", "SN Ib/c": "#E100FF",
    "SLSN-I": "#3DC566ED",
}
TYPE_LABEL = {
    "SN Ic-BL": "Ic-BL", "SN Ic-BL?": "Ic-BL?", "SN Ic": "Ic",
    "SN Ib": "Ib", "SN Ib/c": "Ib/c", "SLSN-I": "SLSN-I",
}
TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]

COLOR_SCAN_PHASES = np.arange(-10.0, 10.0001, 2.5)   # must match measure_population


def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    return pd.read_csv(CSV)


def _cdf_figure(df, col, xlabel, title, out, invert_x=False):
    sub = df[np.isfinite(df[col])]
    if sub.empty:
        print(f"no finite {col}; skipping {title}"); return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for t in TYPE_ORDER:
        v = pd.to_numeric(sub.loc[sub["type"] == t, col], errors="coerce").dropna().values
        if v.size < 2:
            continue
        x = np.sort(v)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=TYPE_COLOR[t], lw=2.0,
                label=f"{TYPE_LABEL[t]} (n={v.size})")
    ax.set_xlabel(xlabel); ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(title)
    ax.legend(fontsize=10, title="type", loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} events)")

def cdf_colorevol(df, sn_type, cmap_name="viridis"):
    """For ONE SN type, draw a CDF of g-r at each scanned epoch (-10..+10 d),
       coloured by phase via a sequential colormap + colorbar (days)."""
    cols = [f"color_p{p:+.1f}" for p in COLOR_SCAN_PHASES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing scan columns {missing[:3]}... "
                         f"re-run measure_population.py with COLOR_SCAN_PHASES")
    sub = df[df["type"] == sn_type]
    if sub.empty:
        print(f"no events of type {sn_type}; skipping"); return
 
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=COLOR_SCAN_PHASES.min(), vmax=COLOR_SCAN_PHASES.max())
 
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n_drawn = 0
    for p, col in zip(COLOR_SCAN_PHASES, cols):
        v = pd.to_numeric(sub[col], errors="coerce").dropna().values
        if v.size < 3:                                  # skip epochs with too few events
            continue
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=cmap(norm(p)), lw=2.0)
        n_drawn += 1
    if n_drawn == 0:
        print(f"{sn_type}: no epoch with >=3 events; skipping"); plt.close(fig); return
 
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("phase relative to r-peak [rest-frame days]")
 
    lab = TYPE_LABEL.get(sn_type, sn_type)
    ax.set_xlabel("colour  g - r  [mag]"); ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Colour evolution CDF  --  {lab}  ({len(sub)} events)")
    ax.grid(alpha=0.2)
    tag = lab.replace("/", "")
    out = os.path.join(OUTDIR, f"cdf_colorevol_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({n_drawn} epochs drawn)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", required=True,
                    choices=["duration", "color", "luminosity", "dtpeak", "rise", "all", "colorevol"])
    
    ap.add_argument("--type", nargs="+", default=None, help="SN type(s) for colorevol, e.g. 'SN Ib' 'SN Ic'")

    ap.add_argument("--band", default="brightest", choices=["g", "r", "brightest"])

    ap.add_argument("--cmap", default="viridis", help="colormap for phase (e.g. cool, plasma)")

    ap.add_argument("--color-mode", default="ownpeak",
                    choices=["rpeak", "gpeak", "ownpeak", "10d"])
    
    a = ap.parse_args()
    df = _load()

    if a.plot in ("duration", "all"):
        col = "fwhm_brightest" if a.band == "brightest" else f"fwhm_{a.band}"
        _cdf_figure(df, col,
                    f"rest-frame FWHM [days] ({a.band})",
                    f"Duration CDF ({a.band} band)",
                    os.path.join(OUTDIR, f"cdf_duration_{a.band}.png"))

    if a.plot in ("rise", "all"):
        band = a.band if a.band != "brightest" else "r"    # rise measured per-band; default r
        col = f"rise_{band}"
        _cdf_figure(df, col, f"rest-frame rise time [days] ({band})",
                    f"Rise-time CDF ({band} band)",
                    os.path.join(OUTDIR, f"cdf_rise_{band}.png"))

    if a.plot in ("color", "all"):
        cmap = {"rpeak": ("color_at_rpeak", "g-r at r-peak"),
                "gpeak": ("color_at_gpeak", "g-r at g-peak"),
                "ownpeak": ("color_gr", "g-r own-peak"),
                "10d": ("color_10d", "g-r at +10d")}
        col, lab = cmap[a.color_mode]
        _cdf_figure(df, col,
                    f"colour  {lab}  [mag]",
                    f"Colour CDF ({lab})",
                    os.path.join(OUTDIR, f"cdf_color_{a.color_mode}.png"))

    if a.plot in ("luminosity", "all"):
        _cdf_figure(df, "M_rest_g",
                    "rest-frame g peak absolute magnitude (K-corrected)",
                    "Luminosity CDF (rest-frame g)",
                    os.path.join(OUTDIR, "cdf_luminosity.png"),
                    invert_x=True)

    if a.plot in ("dtpeak", "all"):
        _cdf_figure(df, "dt_peak_g_minus_r",
                    "peak-time separation  t_g - t_r  [days]",
                    "Peak-separation CDF (g - r)",
                    os.path.join(OUTDIR, "cdf_dtpeak.png"))

    if a.plot == "colorevol":
      types = a.type if a.type else TYPE_ORDER
      for t in types:
          cdf_colorevol(df, t, cmap_name=a.cmap)
if __name__ == "__main__":
    main()