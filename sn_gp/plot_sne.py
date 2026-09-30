"""
plot_sne.py  --  one general plotting tool for the SE-SNe population.
Reads fit/population_measurements.csv. Handles four plot KINDS on any
PARAMETER(s), for any SN type(s).

GRAMMAR
    --kind {scatter,hist,cdf,colorevol}
    --x PARAM              (all kinds; the quantity to plot)
    --y PARAM              (scatter only; the second quantity)
    --type T [T ...]       (one or more types, or "all"; default all)
    --limits               (scatter/cdf: overlay constraining lower limits, where defined)
    --bins N               (hist: bin count, default 25)
    --density              (hist: normalise to density)
    --cmap NAME            (colorevol: phase colormap, default viridis)
    --max-magerr X         (scatter: drop points with magnitude error > X)

EXAMPLES
    # duration-luminosity scatter (your headline plot)
    python plot_sne.py --kind scatter --x fwhm_r --y M_rest_g --limits

    # rise vs total duration
    python plot_sne.py --kind scatter --x rise_r --y fwhm_r

    # colour vs luminosity, only Ib and Ic
    python plot_sne.py --kind scatter --x color_10d --y M_rest_g --type "SN Ib" "SN Ic"

    # any CDF
    python plot_sne.py --kind cdf --x fwhm_r
    python plot_sne.py --kind cdf --x M_rest_g
    python plot_sne.py --kind cdf --x color_10d

    # any histogram
    python plot_sne.py --kind hist --x dt_peak_g_minus_r
    python plot_sne.py --kind hist --x color_at_rpeak --type "SN Ib" "SN Ic"

    # colour-evolution CDF (phase scan), per class
    python plot_sne.py --kind colorevol --type "SN Ic"
    python plot_sne.py --kind colorevol                     # all classes, one fig each

    # list available parameters
    python plot_sne.py --list

    python plot_sne.py --kind colortrack                          # all classes, one plot
    python plot_sne.py --kind colortrack --type "SN Ib"           # one class
    python plot_sne.py --kind colortrack --separate               # one figure per class
    python plot_sne.py --kind colortrack --alpha 0.05             # fainter tracks 

    python plot_sne.py --kind colortrack                        # mag stages (default)
    python plot_sne.py --kind colortrack --axis days            # phase, -10 to +10 d
    python plot_sne.py --kind colortrack --axis days --separate
    python plot_sne.py --kind colortrack --axis days --type "SN Ib" "SN Ic"
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
from matplotlib.patches import Ellipse


plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "legend.title_fontsize": 13,
})

import config

CSV = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "population_plots")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------ style
TYPE_STYLE = {
    "SN Ic-BL":  dict(marker="s", color="#FF0000", label="Ic-BL"),
    # "SN Ic-BL?": dict(marker="s", color="#FF0000", label="Ic-BL"),   # shared label
    "SN Ic":     dict(marker="o", color="#3300FF", label="Ic"),
    "SN Ib":     dict(marker="^", color="#000000", label="Ib"),
    # "SN Ib/c":   dict(marker="D", color="#E100FF", label="Ib/c"),
    "SLSN-I":    dict(marker="*", color="#00C853", label="SLSN-I"),
}
LINESTYLE = {
    "SN Ic": "-", "SN Ib": "--", 
    # "SN Ib/c": ":",
    "SN Ic-BL": "-", "SLSN-I": "-",
}
TYPE_ORDER = ["SN Ic", "SN Ib", 
            #   "SN Ib/c", 
              "SN Ic-BL", "SLSN-I"]


def _style(t): return TYPE_STYLE.get(str(t), dict(marker="x", color="grey", label=str(t)))
def _color(t): return _style(t)["color"]
def _label(t): return _style(t)["label"]

CONSTRAINING_DAYS = 16.0
FAST_RISE_DAYS = 8.0
MAG_REF_LINE = -17.8                       # ceiling line drawn on magnitude axes
COLOR_SCAN_PHASES = np.arange(-10.0, 10.0001, 2.5)
COLOR_MAG_LEVELS = np.round(np.arange(-1.0, 1.5001, 0.1), 1) + 0.0
# ------------------------------------------------------------------ parameter registry
# Each parameter: column, human label, is_mag (invert axis + draw ref line),
# and optional lower-limit companions (ll_col, lim_col) keyed by band where relevant.
# rise_col for the FAST_RISE exception on limits.
def _P(col, label, is_mag=False, ll=None, lim=None, rise=None):
    return dict(col=col, label=label, is_mag=is_mag, ll=ll, lim=lim, rise=rise)

PARAMS = {
    # durations (rest-frame) -- these have lower limits
    "fwhm_r":        _P("fwhm_r", "rest-frame FWHM r [d]", ll="fwhm_r_ll", lim="r_is_limit", rise="rise_r"),
    "fwhm_g":        _P("fwhm_g", "rest-frame FWHM g [d]", ll="fwhm_g_ll", lim="g_is_limit", rise="rise_g"),
    "fwhm_brightest":_P("fwhm_brightest", "rest-frame FWHM [d]", ll="fwhm_brightest_ll", lim="brightest_is_limit"),
    "rise_r":        _P("rise_r", "rest-frame rise r [d]"),
    "rise_g":        _P("rise_g", "rest-frame rise g [d]"),
    "fade_r":        _P("fade_r", "rest-frame fade r [d]"),
    "fade_g":        _P("fade_g", "rest-frame fade g [d]"),
    "rise_frac_g": _P("rise_frac_g", "rest-frame rise g / FWHM g [d/d]"),
    "rise_frac_r": _P("rise_frac_r", "rest-frame rise r / FWHM r [d/d]"),
    "fade_frac_g": _P("fade_frac_g", "rest-frame fade g / FWHM g [d/d]"),
    "fade_frac_r": _P("fade_frac_r", "rest-frame fade r / FWHM r [d/d]"),
    # luminosities (magnitudes -> invert, draw ceiling line)
    "M_rest_g":      _P("M_rest_g", "rest-frame g peak abs mag (K-corr)", is_mag=True),
    "M_r":           _P("M_r", "r-band peak abs mag (obs)", is_mag=True),
    "M_g":           _P("M_g", "g-band peak abs mag (obs)", is_mag=True),
    # colours (observed frame)
    "color_gr":      _P("color_gr", "g-r own-peak [mag]"),
    "color_at_rpeak":_P("color_at_rpeak", "g-r at r-peak [mag]"),
    "color_at_gpeak":_P("color_at_gpeak", "g-r at g-peak [mag]"),
    "color_10d":     _P("color_10d", "g-r at +10d [mag]"),
    # timing
    "dt_peak_g_minus_r": _P("dt_peak_g_minus_r", r"$\Delta t_{gr}$ [d]"),
        **{f"color_m{d:+.1f}": _P(f"color_m{d:+.1f}",
        f"g-r at {abs(d):.1f} mag {'rise' if d < 0 else 'decline' if d > 0 else 'peak'}")
       for d in COLOR_MAG_LEVELS},
    "tail_slope_r":  _P("tail_slope_r", "tail slope r [mag/d, +30..60d]"),
    "tail_slope_g":  _P("tail_slope_g", "tail slope g [mag/d, +30..60d]"),
    "dm15_r":        _P("dm15_r", "Δm15 r [mag]"),
    "dm15_g":        _P("dm15_g", "Δm15 g [mag]"),
    "fwhm_ratio_gr": _P("fwhm_ratio_gr", "FWHM ratio g/r"),
    "color_rate":    _P("color_rate", "d(g-r)/dt [mag/d, 0..+10d]"),
    "dcolor_pre":      _P("dcolor_pre",      r"$\Delta(g-r)$ peak $-$ ($-10$d) [mag]"),
    "dcolor_post":     _P("dcolor_post",     r"$\Delta(g-r)$ (+10$d) $-$ peak [mag]"),
    "dcolor_full":     _P("dcolor_full",     r"$\Delta(g-r)$ (+10$d) $-$ ($-10$d) [mag]"),
    "dcolor_pre_abs":  _P("dcolor_pre_abs",  r"$|\Delta(g-r)|$ pre-peak [mag]"),
    "dcolor_post_abs": _P("dcolor_post_abs", r"$|\Delta(g-r)|$ post-peak [mag]"),
    "dcolor_full_abs": _P("dcolor_full_abs", r"$|\Delta(g-r)|$ ±10$d [mag]"),
}

def _resolve(name):
    if name in PARAMS:
        return PARAMS[name]
    # allow raw column names not in the registry (plain label, no special handling)
    return _P(name, name)

# ------------------------------------------------------------------ data
MORPH = os.path.join(config.FIT_ROOT, "morphology_review.csv")
MORPH_COLS = ["shape", "features", "below15", "very_bad", "notes"]

def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    df = pd.read_csv(CSV)
    if os.path.exists(MORPH):
        m = pd.read_csv(MORPH, dtype=str).fillna("")
        keep = ["ZTFID"] + [c for c in MORPH_COLS if c in m.columns]
        df = df.merge(m[keep], on="ZTFID", how="left")
        for c in MORPH_COLS:
            if c in df.columns:
                df[c] = df[c].fillna("")
    else:
        print(f"   note: no {MORPH}; morphology cuts unavailable")
    return df

def _types_arg(types):
    if not types or (len(types) == 1 and types[0].lower() == "all"):
        return TYPE_ORDER
    return types

def _finite(df, col):
    return pd.to_numeric(df[col], errors="coerce")

EXCLUDE_FILE = os.path.join(config.FIT_ROOT, "exclude_events.txt")

def _apply_exclusions(df, extra=None):
    """Drop events listed in exclude_events.txt and/or given via --exclude.
       Matching is case-insensitive and substring-based, so 'aadaxfn' matches
       'ZTF23aadaxfn'."""
    if "very_bad" in df.columns:
        vb = df["very_bad"].astype(str).str.lower() == "yes"
        if vb.any():
            print(f"   excluded {int(vb.sum())} 'very bad' event(s)")
            df = df[~vb].copy()
    pats = []
    if os.path.exists(EXCLUDE_FILE):
        with open(EXCLUDE_FILE) as f:
            pats += [ln.strip() for ln in f
                     if ln.strip() and not ln.strip().startswith("#")]
    if extra:
        pats += list(extra)
    if not pats:
        return df
    ids = df["ZTFID"].astype(str).str.lower()
    drop = pd.Series(False, index=df.index)
    for p in pats:
        hit = ids.str.contains(p.strip().lower(), regex=False)
        if not hit.any():
            print(f"   exclude: no match for '{p}'")
        drop |= hit
    if drop.any():
        print(f"   excluded {int(drop.sum())} event(s): "
              f"{sorted(df.loc[drop, 'ZTFID'].astype(str))}")
    return df[~drop].copy()

# ------------------------------------------------------------------ plot kinds
def plot_scatter(df, xp, yp, types, limits=False, max_magerr=None):
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"])
    meas = x.notna() & y.notna()
    # scatter limits only make sense if the x-parameter has a lower-limit column
    if limits and not xp.get("lim"):
        print(f"   note: '{xp['col']}' has no lower-limit column; ignoring --limits")
        limits = False
    if limits:
        meas &= ~df[xp["lim"]].fillna(False)          # measured points only in main pass
    # magnitude-error cap (only if y is a magnitude with an err column)
    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        too_big = _finite(df, yerr_col) > max_magerr
        n = int((meas & too_big).sum()); meas &= ~too_big
        print(f"   error cap {max_magerr}: dropped {n} points")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    seen = set()
    for _, r in df[meas & df["type"].isin(types)].iterrows():
        s = _style(r["type"]); lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])
        ye = (r[yerr_col] if (yp["is_mag"] and yerr_col in df.columns
                              and np.isfinite(r.get(yerr_col, np.nan))) else None)
        ax.errorbar(r[xp["col"]], r[yp["col"]], yerr=ye, fmt=s["marker"],
                    color=s["color"], ms=6, mec="k", mew=0.4, alpha=0.85,
                    elinewidth=0.7, capsize=1.5, label=lbl)

    # lower-limit hollow markers (x-parameter limits: FWHM etc.)
    if limits:
        lim = df[xp["lim"]].fillna(False) & _finite(df, xp["ll"]).notna() & y.notna()
        lim &= (_finite(df, xp["ll"]) >= CONSTRAINING_DAYS)
        if xp.get("rise") and xp["rise"] in df.columns:
            fast = _finite(df, xp["rise"]).notna() & (_finite(df, xp["rise"]) < FAST_RISE_DAYS)
            lim = df[xp["lim"]].fillna(False) & _finite(df, xp["ll"]).notna() & y.notna() & \
                  ((_finite(df, xp["ll"]) >= CONSTRAINING_DAYS) | fast)
        for _, r in df[lim & df["type"].isin(types)].iterrows():
            s = _style(r["type"])
            ax.errorbar(r[xp["ll"]], r[yp["col"]], fmt=s["marker"], mfc="none",
                        mec=s["color"], ms=7, mew=1.1, alpha=0.9)
        print(f"   overlaid {int((lim & df['type'].isin(types)).sum())} lower-limit markers")

    ax.set_xlabel(xp["label"]); ax.set_ylabel(yp["label"])
    if xp["is_mag"]: ax.invert_xaxis()
    if yp["is_mag"]:
        ax.invert_yaxis()
        ax.axhline(MAG_REF_LINE, color="k", ls="--", lw=1.2, alpha=0.7)
        ax.text(0.99, MAG_REF_LINE, f" {MAG_REF_LINE}", va="bottom", ha="right",
                fontsize=9, color="k", transform=ax.get_yaxis_transform())
    # ax.set_title(f"{yp['label']}  vs  {xp['label']}")
    ax.legend(fontsize=11, title="type", loc="best"); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"scatter_{yp['col']}_vs_{xp['col']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({int((meas & df['type'].isin(types)).sum())} points)")

def plot_scatter3(df, xp, yp, cp, types, cmap_name="viridis",
                  vmin=None, vmax=None, max_magerr=None):
    """Scatter of y vs x with marker COLOUR encoding a third parameter (cp),
       shown via a colourbar. Marker SHAPE still encodes SN type."""
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"]); c = _finite(df, cp["col"])
    meas = x.notna() & y.notna() & c.notna() & df["type"].isin(types)

    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        too_big = _finite(df, yerr_col) > max_magerr
        n = int((meas & too_big).sum()); meas &= ~too_big
        print(f"   error cap {max_magerr}: dropped {n} points")

    sub = df[meas]
    if sub.empty:
        print(f"no points with finite {xp['col']}, {yp['col']}, {cp['col']}"); return

    cvals = _finite(sub, cp["col"])
    if vmin is None: vmin = float(np.nanpercentile(cvals, 2))
    if vmax is None: vmax = float(np.nanpercentile(cvals, 98))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for t in TYPE_ORDER:                       # one scatter call per type -> shape legend
        if t not in types:
            continue
        s = sub[sub["type"] == t]
        if s.empty:
            continue
        mk = _style(t)["marker"]; lab = _label(t)
        ax.scatter(_finite(s, xp["col"]), _finite(s, yp["col"]),
                   c=_finite(s, cp["col"]), cmap=cmap, norm=norm,
                   marker=mk, s=50, edgecolor="k", linewidth=0.5, zorder=3, alpha = 0.85)
        # hollow, colourless proxy for the legend (shape only)
        handles.append(plt.Line2D([], [], linestyle="none", marker=mk,
                                  markerfacecolor="none", markeredgecolor="black",
                                  markeredgewidth=1.0, markersize=9,
                                  label=f"{lab} (n={len(s)})"))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label(cp["label"])

    ax.set_xlabel(xp["label"]); ax.set_ylabel(yp["label"])
    if xp["is_mag"]:
        ax.invert_xaxis()
    if yp["is_mag"]:
        ax.invert_yaxis()
        ax.axhline(MAG_REF_LINE, color="k", ls="--", lw=1.2, alpha=0.6)
    # ax.axhline(0, color="grey", ls=":", lw=1.0, alpha=0.6)
    # ax.set_title(f"{yp['label']}  vs  {xp['label']}\ncoloured by {cp['label']}",
    #              fontsize=13)
    ax.legend(handles=handles, fontsize=8, title="type", loc="best"); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR,
                       f"scatter3_{yp['col']}_vs_{xp['col']}_c_{cp['col']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} points)")

def plot_hist(df, xp, types, bins=25, density=False):
    x = _finite(df, xp["col"])
    # drop lower limits if this parameter has them
    if xp.get("lim"):
        mask = x.notna() & ~df[xp["lim"]].fillna(False)
    else:
        mask = x.notna()
    sub = df[mask]
    if sub.empty:
        print(f"no finite {xp['col']}; skipping"); return
    vals = _finite(sub, xp["col"])
    edges = np.histogram_bin_edges(vals.dropna().values, bins=bins)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for t in TYPE_ORDER:
        if t not in types: continue
        d = _finite(sub[sub["type"] == t], xp["col"]).dropna().values
        if d.size == 0: continue
        lab = _label(t); lbl = lab if lab not in seen else None; seen.add(lab)
        ax.hist(d, bins=edges, histtype="step", color=_color(t),
                linestyle=LINESTYLE[t], linewidth=2.2, density=density, label=lbl)
    if xp["is_mag"]: ax.invert_xaxis()
    ax.set_xlabel(xp["label"]); ax.set_ylabel("density" if density else "number of events")
    ax.set_title(f"{xp['label']}  --  histogram")
    ax.legend(fontsize=11, title="type"); ax.grid(alpha=0.2, axis="y")
    out = os.path.join(OUTDIR, f"hist_{xp['col']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} events)")


def plot_cdf(df, xp, types):
    if xp.get("lim"):
        sub = df[_finite(df, xp["col"]).notna() & ~df[xp["lim"]].fillna(False)]
    else:
        sub = df[_finite(df, xp["col"]).notna()]
    if sub.empty:
        print(f"no finite {xp['col']}; skipping"); return
    fig, ax = plt.subplots(figsize=(5, 6))
    for t in TYPE_ORDER:
        if t not in types: continue
        v = _finite(sub[sub["type"] == t], xp["col"]).dropna().values
        if v.size < 2: continue
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=_color(t), lw=2.0,
                label=f"{_label(t)}")
    ax.set_ylim(0, 1.02); ax.set_ylabel("cumulative fraction"); ax.set_xlabel(xp["label"])
    if xp["is_mag"]:
        ax.invert_xaxis()
        ax.axvline(MAG_REF_LINE, color="k", ls="--", lw=1.2, alpha=0.7)
    # ax.set_title(f"{xp['label']}  --  CDF")
    ax.legend(fontsize=11, title="type", loc="best"); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"cdf_{xp['col']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} events)")


def _scan_xlim(df, cols, pad_frac=0.03):
    """Global x-range across ALL types and ALL scan columns, so per-class
       figures share one scale and are directly comparable."""
    vals = pd.concat([_finite(df, c) for c in cols if c in df.columns]).dropna()
    if vals.empty:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = pad_frac * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad

def plot_magevol(df, sn_type, cmap_name="viridis"):
    """For ONE type, CDF of g-r at each r-band luminosity stage, coloured by stage."""
    cols = [f"color_m{d:+.1f}" for d in COLOR_MAG_LEVELS]
    xlim = _scan_xlim(df, cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing {missing[0]}; re-run measure_population.py "
                         f"with COLOR_MAG_LEVELS")
    sub = df[df["type"] == sn_type]
    if sub.empty:
        print(f"no events of type {sn_type}; skipping"); return
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=COLOR_MAG_LEVELS.min(), vmax=COLOR_MAG_LEVELS.max())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_drawn = 0
    for d, col in zip(COLOR_MAG_LEVELS, cols):
        v = _finite(sub, col).dropna().values
        if v.size < 3: continue
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=cmap(norm(d)), lw=2.0)
        n_drawn += 1
    if n_drawn == 0:
        print(f"{sn_type}: no stage with >=3 events; skipping"); plt.close(fig); return
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax)
    lab = _label(sn_type)
    ax.set_xlabel("colour  g - r  [mag]"); ax.set_ylabel("cumulative fraction")
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Colour vs luminosity stage -- {lab} ({len(sub)} events)")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"magevol_{lab.replace('/', '')}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({n_drawn} stages)")

def plot_colorevol(df, sn_type, cmap_name="viridis"):
    cols = [f"color_p{p:+.1f}" for p in COLOR_SCAN_PHASES]
    xlim = _scan_xlim(df, cols)          # from ALL types, not just this one
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing scan columns (e.g. {missing[0]}); re-run "
                         f"measure_population.py with COLOR_SCAN_PHASES")
    sub = df[df["type"] == sn_type]
    if sub.empty:
        print(f"no events of type {sn_type}; skipping"); return
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=COLOR_SCAN_PHASES.min(), vmax=COLOR_SCAN_PHASES.max())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_drawn = 0
    for p, col in zip(COLOR_SCAN_PHASES, cols):
        v = _finite(sub, col).dropna().values
        if v.size < 3: continue
        x = np.sort(v); y = np.arange(1, x.size + 1) / x.size
        ax.plot(np.concatenate([[x[0]], x]), np.concatenate([[0], y]),
                drawstyle="steps-post", color=cmap(norm(p)), lw=2.0)
        n_drawn += 1
    if n_drawn == 0:
        print(f"{sn_type}: no epoch with >=3 events; skipping"); plt.close(fig); return
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax)
    lab = _label(sn_type)
    ax.set_xlabel("colour  g - r  [mag]"); ax.set_ylabel("cumulative fraction")
    ax.set_xlim(xlim); ax.set_ylim(0, 1.02); ax.set_title(f"Colour evolution CDF -- {lab} ({len(sub)} events)")
    ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"colorevol_{lab.replace('/', '')}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({n_drawn} epochs)")

def _stage_cols(df, prefix):
    """Find color_m* (mag stages) or color_p* (phase) columns present in df."""
    found = []
    for c in df.columns:
        if c.startswith(prefix):
            try:
                found.append((float(c[len(prefix):]), c))
            except ValueError:
                pass
    if not found:
        raise SystemExit(f"no {prefix}* columns in the CSV -- re-run "
                         f"measure_population.py")
    found.sort()
    return np.array([f[0] for f in found]), [f[1] for f in found]

def _colortrack_subset(df):
    """Events usable for colour tracks: data exist below r-peak +1.5 mag on the
       fading side, AND the g and r shapes agree (no 'colorodd' flag)."""
    if "below15" not in df.columns:
        print("   no morphology tags; colour tracks use all events")
        return df
    n0 = len(df)
    ok = df["below15"].astype(str).str.lower() == "yes"
    ok &= ~df["features"].astype(str).str.contains("colorodd", case=False, na=False)
    print(f"   morphology cut: {int(ok.sum())}/{n0} events "
          f"(below +1.5 mag, g/r shapes agree)")
    return df[ok].copy()

def plot_colortrack(df, types, alpha=0.12, show_median=True, separate=False,
                    axis="mag", normalise=None, ref_stage=0.0, morph_cut=True):
    """One faint g-r track per event, plus a bold per-class median.
       axis='mag'  -> vs r-band luminosity stage (color_m* columns)
       axis='days' -> vs rest-frame phase from r-peak (color_p* columns)
       normalise:  None    -> raw colours
                   'event' -> subtract each event's own colour at ref_stage
                   'class' -> subtract each class's median colour at ref_stage
       Normalising removes the vertical offset (a constant offset is what host
       extinction produces), so what remains is the SHAPE of the colour
       evolution -- which extinction cannot change."""
    if morph_cut:
        df = _colortrack_subset(df)
    if axis == "mag":
        levels, cols = _stage_cols(df, "color_m")
        xlabel = "r-band stage [mag from peak]   (-ve = rise, +ve = decline)"
        titlebit = "luminosity stage"
    elif axis == "days":
        levels, cols = _stage_cols(df, "color_p")
        xlabel = "phase relative to r-peak [rest-frame days]"
        titlebit = "phase"
    else:
        raise SystemExit("axis must be 'mag' or 'days'")

    if normalise not in (None, "event", "class"):
        raise SystemExit("normalise must be 'event', 'class' or omitted")
    iref = int(np.argmin(np.abs(levels - ref_stage)))
    ylabel = ("colour  g - r  [mag]" if not normalise
              else f"$\\Delta$(g - r) relative to stage {levels[iref]:+.1f}  [mag]")

    groups = [[t] for t in types] if separate else [list(types)]
    for grp in groups:
        fig, ax = plt.subplots(figsize=(8, 6))
        seen = set()
        for t in TYPE_ORDER:
            if t not in grp:
                continue
            sub = df[df["type"] == t]
            if sub.empty:
                continue
            vals = sub[cols].apply(pd.to_numeric, errors="coerce")

            # ---- remove the vertical offset (extinction test) ----
            if normalise == "event":
                vals = vals.sub(vals.iloc[:, iref], axis=0)
            elif normalise == "class":
                vals = vals - vals.iloc[:, iref].median(skipna=True)

            n_tracks = 0
            for _, row in vals.iterrows():
                y = row.values.astype(float)
                if np.isfinite(y).sum() < 2:
                    continue
                ax.plot(levels, y, color=_color(t), lw=1.0, alpha=alpha, zorder=1)
                n_tracks += 1
            if n_tracks == 0:
                continue
            if show_median:
                med = vals.median(axis=0, skipna=True).values.astype(float)
                lab = _label(t)
                ax.plot(levels, med, color=_color(t), lw=3.0, alpha=0.95, zorder=3,
                        label=None if lab in seen else f"{lab} (n={n_tracks})")
                seen.add(lab)

                # shape summary, extinction-independent
                sel = (np.abs(levels) <= 0.4) & np.isfinite(med)
                slope = (np.polyfit(levels[sel], med[sel], 1)[0]
                         if sel.sum() >= 3 else np.nan)
                fin = np.isfinite(med)
                if fin.any():
                    ired = int(np.nanargmax(np.where(fin, med, -np.inf)))
                    turn = float(med[ired] - med[fin][-1])   # reddest -> last stage
                    print(f"   {lab:8s} slope@peak={slope:+.3f}  "
                          f"reddest at {levels[ired]:+.1f}  turnaround={turn:+.3f}")

        ax.axvline(levels[iref], color="k", ls=":", lw=1.0, alpha=0.6)
        if normalise:
            ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        title = (_label(grp[0]) if separate else "all classes")
        norm_bit = "" if not normalise else f"  [{normalise}-normalised]"
        # ax.set_title(f"Colour tracks vs {titlebit} -- {title}{norm_bit}")
        if show_median:
            ax.legend(fontsize=11, title="median track", loc="best")
        ax.grid(alpha=0.2)
        tag = _label(grp[0]).replace("/", "") if separate else "all"
        ntag = f"_{normalise}norm" if normalise else ""
        out = os.path.join(OUTDIR, f"colortrack_{axis}_{tag}{ntag}.png")
        fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
        print(f"wrote {out}")

from scipy.stats import gaussian_kde


def _kde_contour(ax, x, y, fracs, color, bw=None, gridsize=200):
    """Unfilled KDE contours enclosing the given fractions of the probability
       mass. Follows the actual point distribution (not a Gaussian ellipse)."""
    if len(x) < 5:
        return False
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    except Exception:
        return False                      # singular covariance / degenerate data

    padx = 0.25 * ((x.max() - x.min()) or 1.0)
    pady = 0.25 * ((y.max() - y.min()) or 1.0)
    xg = np.linspace(x.min() - padx, x.max() + padx, gridsize)
    yg = np.linspace(y.min() - pady, y.max() + pady, gridsize)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

    # density threshold enclosing each fraction: sort densities descending,
    # accumulate mass, take the level where the cumulative reaches the fraction
    cell = (xg[1] - xg[0]) * (yg[1] - yg[0])
    flat = np.sort(ZZ.ravel())[::-1]
    cum = np.cumsum(flat) * cell
    cum /= cum[-1]
    thresh = []
    for f in sorted(fracs, reverse=True):        # larger fraction -> lower level
        idx = min(int(np.searchsorted(cum, f)), len(flat) - 1)
        thresh.append(float(flat[idx]))
    thresh = sorted(set(thresh))                 # contour() needs ascending
    if not thresh:
        return False

    lws = np.linspace(1.3, 2.4, len(thresh))     # outer thin -> inner thick
    ax.contour(XX, YY, ZZ, levels=thresh, colors=[color] * len(thresh),
               linewidths=list(lws), zorder=3)
    return True


def plot_scatter_contour(df, xp, yp, types, fracs=(0.68, 0.95), bw=None,
                         show_points=True, max_magerr=None):
    """Scatter with unfilled KDE density contours per class, following the
       actual point distribution."""
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"])
    meas = x.notna() & y.notna() & df["type"].isin(types)

    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        too_big = _finite(df, yerr_col) > max_magerr
        n = int((meas & too_big).sum()); meas &= ~too_big
        print(f"   error cap {max_magerr}: dropped {n} points")

    sub = df[meas]
    if sub.empty:
        print(f"no points with finite {xp['col']} and {yp['col']}"); return

    fig, ax = plt.subplots(figsize=(8, 6.5))
    seen = set()
    for t in TYPE_ORDER:
        if t not in types:
            continue
        s = sub[sub["type"] == t]
        if s.empty:
            continue
        xv = _finite(s, xp["col"]).values.astype(float)
        yv = _finite(s, yp["col"]).values.astype(float)
        col = _color(t); lab = _label(t)

        if show_points:
            ax.scatter(xv, yv, marker=_style(t)["marker"], color=col, s=26,
                       edgecolor="k", linewidth=0.3, alpha=0.5, zorder=2,
                       label=None if lab in seen else f"{lab} (n={len(s)})")
            seen.add(lab)

        ok = _kde_contour(ax, xv, yv, fracs, col, bw=bw)
        if not ok:
            print(f"   {lab}: too few points ({len(s)}) for a KDE contour")

    ax.set_xlabel(xp["label"]); ax.set_ylabel(yp["label"])
    if xp["is_mag"]:
        ax.invert_xaxis()
        ax.axvline(MAG_REF_LINE, color="k", ls="--", lw=1.1, alpha=0.5)
    if yp["is_mag"]:
        ax.invert_yaxis()
        ax.axhline(MAG_REF_LINE, color="k", ls="--", lw=1.1, alpha=0.5)
    lv = ", ".join(f"{int(f*100)}%" for f in sorted(fracs))
    ax.set_title(f"{yp['label']}  vs  {xp['label']}\nKDE contours ({lv})",
                 fontsize=13)
    ax.legend(fontsize=10, title="type", loc="best"); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"contour_{yp['col']}_vs_{xp['col']}.png")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  ({len(sub)} points)")

# ================================================================== HTML
import re

PLOTLY_SYMBOL = {"o": "circle", "^": "triangle-up", "s": "square",
                 "D": "diamond", "*": "star", "x": "x"}
def _psym(t): return PLOTLY_SYMBOL.get(_style(t)["marker"], "circle")

def _plain(lab):
    """matplotlib mathtext -> plain text (plotly has no mathtext by default)."""
    s = str(lab).replace("$", "").replace(r"\Delta", "Δ").replace(r"\mathrm", "")
    s = s.replace(r"\,", " ")
    return re.sub(r"[{}]", "", s)

def _rgba(hexcol, alpha):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

HOVER_HEAD = ("<b>%{customdata[0]}</b><br>type: %{customdata[1]}"
              "<br>z: %{customdata[2]}<br>kernel: %{customdata[3]}"
              "<br>shape: %{customdata[4]}<br>features: %{customdata[5]}")

def _custom(sub):
    n = len(sub)
    def col(c, fmt=None):
        if c not in sub.columns:
            return pd.Series(["--"] * n, index=sub.index).astype(str)
        if fmt:
            return sub[c].map(lambda v: fmt.format(v) if pd.notna(v) else "--").astype(str)
        return sub[c].fillna("--").astype(str)
    return np.stack([col("ZTFID"), sub["type"].map(_label).astype(str),
                     col("z", "{:.4f}"), col("kernel"),
                     col("shape"), col("features")], axis=-1)

def _write_html(fig, name):
    out = os.path.join(OUTDIR, f"{name}.html")
    fig.write_html(out, include_plotlyjs=True)     # self-contained, works offline
    print(f"wrote {out}")

def _axes(fig, xp, yp):
    fig.update_layout(xaxis_title=_plain(xp["label"]),
                      yaxis_title=_plain(yp["label"]),
                      template="simple_white", legend_title="type",
                      width=950, height=720, hovermode="closest")
    if xp["is_mag"]:
        fig.update_xaxes(autorange="reversed")
        if xp["col"] == "M_r":
            fig.add_vline(x=MAG_REF_LINE, line=dict(dash="dash", color="black", width=1))
    if yp["is_mag"]:
        fig.update_yaxes(autorange="reversed")
        if yp["col"] == "M_r":
            fig.add_hline(y=MAG_REF_LINE, line=dict(dash="dash", color="black", width=1))


def plot_scatter_html(df, xp, yp, types, limits=False, max_magerr=None):
    import plotly.graph_objects as go
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"])
    meas = x.notna() & y.notna() & df["type"].isin(types)
    if limits and xp.get("lim"):
        meas &= ~df[xp["lim"]].fillna(False)
    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        meas &= ~(_finite(df, yerr_col) > max_magerr)

    fig = go.Figure()
    for t in TYPE_ORDER:
        if t not in types: continue
        s = df[meas & (df["type"] == t)]
        if s.empty: continue
        err = (dict(type="data", array=_finite(s, yerr_col), visible=True)
               if (yp["is_mag"] and yerr_col in s.columns) else None)
        fig.add_trace(go.Scatter(
            x=_finite(s, xp["col"]), y=_finite(s, yp["col"]), mode="markers",
            name=f"{_label(t)} (n={len(s)})", legendgroup=t, error_y=err,
            marker=dict(size=9, color=_color(t), symbol=_psym(t),
                        line=dict(width=0.6, color="black")),
            customdata=_custom(s),
            hovertemplate=HOVER_HEAD +
                f"<br>{_plain(xp['label'])}: " + "%{x:.3f}" +
                f"<br>{_plain(yp['label'])}: " + "%{y:.3f}<extra></extra>"))

    # hollow lower limits
    if limits and xp.get("lim"):
        ll = df[xp["lim"]].fillna(False) & _finite(df, xp["ll"]).notna() & y.notna()
        ll &= (_finite(df, xp["ll"]) >= CONSTRAINING_DAYS)
        for t in TYPE_ORDER:
            if t not in types: continue
            s = df[ll & (df["type"] == t)]
            if s.empty: continue
            fig.add_trace(go.Scatter(
                x=_finite(s, xp["ll"]), y=_finite(s, yp["col"]), mode="markers",
                name=f"{_label(t)} limit", legendgroup=t, showlegend=False,
                marker=dict(size=10, color=_color(t), symbol=_psym(t) + "-open",
                            line=dict(width=1.4, color=_color(t))),
                customdata=_custom(s),
                hovertemplate=HOVER_HEAD + "<br>lower limit: %{x:.1f}<extra></extra>"))
    _axes(fig, xp, yp)
    _write_html(fig, f"scatter_{yp['col']}_vs_{xp['col']}")


def plot_scatter3_html(df, xp, yp, cp, types, cmap_name="viridis",
                       vmin=None, vmax=None, max_magerr=None):
    import plotly.graph_objects as go
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"]); c = _finite(df, cp["col"])
    meas = x.notna() & y.notna() & c.notna() & df["type"].isin(types)
    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        meas &= ~(_finite(df, yerr_col) > max_magerr)
    sub = df[meas]
    if sub.empty:
        print("no points"); return
    cv = _finite(sub, cp["col"])
    lo = float(np.nanpercentile(cv, 2)) if vmin is None else vmin
    hi = float(np.nanpercentile(cv, 98)) if vmax is None else vmax

    fig = go.Figure()
    for i, t in enumerate([t for t in TYPE_ORDER if t in types]):
        s = sub[sub["type"] == t]
        if s.empty: continue
        fig.add_trace(go.Scatter(
            x=_finite(s, xp["col"]), y=_finite(s, yp["col"]), mode="markers",
            name=f"{_label(t)} (n={len(s)})",
            marker=dict(size=11, symbol=_psym(t),
                        color=_finite(s, cp["col"]), colorscale=cmap_name,
                        cmin=lo, cmax=hi, line=dict(width=0.6, color="black"),
                        showscale=(i == 0),
                        colorbar=dict(title=_plain(cp["label"]))),
            customdata=np.concatenate(
                [_custom(s), _finite(s, cp["col"]).values.reshape(-1, 1).astype(str)],
                axis=1),
            hovertemplate=HOVER_HEAD +
                f"<br>{_plain(xp['label'])}: " + "%{x:.3f}" +
                f"<br>{_plain(yp['label'])}: " + "%{y:.3f}" +
                f"<br>{_plain(cp['label'])}: " + "%{customdata[6]}<extra></extra>"))
    _axes(fig, xp, yp)
    _write_html(fig, f"scatter3_{yp['col']}_vs_{xp['col']}_c_{cp['col']}")


def plot_contour_html(df, xp, yp, types, fracs=(0.68, 0.95), bw=None,
                      show_points=True, max_magerr=None):
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    x = _finite(df, xp["col"]); y = _finite(df, yp["col"])
    meas = x.notna() & y.notna() & df["type"].isin(types)
    yerr_col = yp["col"] + "_err"
    if max_magerr is not None and yp["is_mag"] and yerr_col in df.columns:
        meas &= ~(_finite(df, yerr_col) > max_magerr)
    sub = df[meas]

    fig = go.Figure()
    for t in TYPE_ORDER:
        if t not in types: continue
        s = sub[sub["type"] == t]
        if s.empty: continue
        xv = _finite(s, xp["col"]).values.astype(float)
        yv = _finite(s, yp["col"]).values.astype(float)
        if len(xv) >= 5:
            try:
                kde = gaussian_kde(np.vstack([xv, yv]), bw_method=bw)
                px = 0.25 * ((xv.max() - xv.min()) or 1.0)
                py = 0.25 * ((yv.max() - yv.min()) or 1.0)
                xg = np.linspace(xv.min() - px, xv.max() + px, 160)
                yg = np.linspace(yv.min() - py, yv.max() + py, 160)
                XX, YY = np.meshgrid(xg, yg)
                ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                cell = (xg[1] - xg[0]) * (yg[1] - yg[0])
                flat = np.sort(ZZ.ravel())[::-1]
                cum = np.cumsum(flat) * cell; cum /= cum[-1]
                for f in sorted(fracs, reverse=True):
                    lv = float(flat[min(int(np.searchsorted(cum, f)), len(flat) - 1)])
                    fig.add_trace(go.Contour(
                        x=xg, y=yg, z=ZZ, showscale=False, hoverinfo="skip",
                        legendgroup=t, showlegend=False,
                        contours=dict(start=lv, end=lv, size=0, coloring="lines"),
                        line=dict(color=_color(t), width=2)))
            except Exception:
                pass
        if show_points:
            fig.add_trace(go.Scatter(
                x=xv, y=yv, mode="markers", name=f"{_label(t)} (n={len(s)})",
                legendgroup=t,
                marker=dict(size=7, color=_color(t), symbol=_psym(t), opacity=0.6,
                            line=dict(width=0.4, color="black")),
                customdata=_custom(s),
                hovertemplate=HOVER_HEAD + "<br>%{x:.3f}, %{y:.3f}<extra></extra>"))
    _axes(fig, xp, yp)
    _write_html(fig, f"contour_{yp['col']}_vs_{xp['col']}")


def plot_colortrack_html(df, types, alpha=0.2, axis="mag", normalise=None,
                         ref_stage=0.0, morph_cut=True):
    import plotly.graph_objects as go
    if morph_cut:
        df = _colortrack_subset(df)
    prefix = "color_m" if axis == "mag" else "color_p"
    levels, cols = _stage_cols(df, prefix)
    xlab = ("r-band stage [mag from peak] (-ve rise, +ve decline)" if axis == "mag"
            else "phase relative to r-peak [rest-frame days]")
    iref = int(np.argmin(np.abs(levels - ref_stage)))
    ylab = ("colour g-r [mag]" if not normalise
            else f"Δ(g-r) relative to stage {levels[iref]:+.1f} [mag]")

    fig = go.Figure()
    for t in TYPE_ORDER:
        if t not in types: continue
        sub = df[df["type"] == t]
        if sub.empty: continue
        vals = sub[cols].apply(pd.to_numeric, errors="coerce")
        if normalise == "event":
            vals = vals.sub(vals.iloc[:, iref], axis=0)
        elif normalise == "class":
            vals = vals - vals.iloc[:, iref].median(skipna=True)

        xs, ys, ids, n_tr = [], [], [], 0
        for (_, row), zid in zip(vals.iterrows(), sub["ZTFID"].astype(str)):
            yv = row.values.astype(float); ok = np.isfinite(yv)
            if ok.sum() < 2: continue
            xs.extend(levels[ok]); ys.extend(yv[ok]); ids.extend([zid] * int(ok.sum()))
            xs.append(None); ys.append(None); ids.append(None)
            n_tr += 1
        if n_tr:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", legendgroup=t, showlegend=False,
                name=f"{_label(t)} tracks",
                line=dict(color=_rgba(_color(t), alpha), width=1),
                customdata=ids,
                hovertemplate="<b>%{customdata}</b><br>stage %{x:+.1f}"
                              "<br>g-r %{y:.3f}<extra></extra>"))
            med = vals.median(axis=0, skipna=True).values.astype(float)
            fig.add_trace(go.Scatter(
                x=levels, y=med, mode="lines", legendgroup=t,
                name=f"{_label(t)} (n={n_tr})",
                line=dict(color=_color(t), width=4),
                hovertemplate=f"{_label(t)} median<br>stage %{{x:+.1f}}"
                              "<br>g-r %{y:.3f}<extra></extra>"))
    fig.add_vline(x=levels[iref], line=dict(dash="dot", color="black", width=1))
    if normalise:
        fig.add_hline(y=0, line=dict(dash="dot", color="black", width=1))
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, template="simple_white",
                      legend_title="type", width=950, height=720, hovermode="closest")
    ntag = f"_{normalise}norm" if normalise else ""
    tag = _label(types[0]).replace("/", "") if len(types) == 1 else "all"
    _write_html(fig, f"colortrack_{axis}_{tag}{ntag}")

def morph_stats(df, types, plot=True):
    """Cross-tabulate primary shape and features against SN type."""
    if "shape" not in df.columns:
        raise SystemExit("no morphology_review.csv found")
    sub = df[df["type"].isin(types)].copy()
    sub["cls"] = sub["type"].map(lambda t: _label(t))     # merges Ic-BL?
    order = []
    for t in TYPE_ORDER:
        if _label(t) not in order and (sub["cls"] == _label(t)).any():
            order.append(_label(t))

    # ---- shape x class ----
    sh = sub[sub["shape"].astype(str) != ""]
    ct = pd.crosstab(sh["shape"], sh["cls"]).reindex(columns=order, fill_value=0)
    print("\n=== primary shape vs class (counts) ===")
    print(ct.to_string())
    print(f"\ntotal tagged: {len(sh)}")
    pct_s = 100 * ct / ct.sum(axis=0).replace(0, np.nan)
    print("\n=== primary shape vs class (% of each class) ===")
    print(pct_s.round(1).to_string())

    # ---- features x class ----
    feats = sorted({f for s in sub["features"].astype(str)
                    for f in s.split(";") if f})
    rows = {}
    for f in feats:
        has = sub["features"].astype(str).str.contains(f, case=False, na=False)
        rows[f] = {c: int((has & (sub["cls"] == c)).sum()) for c in order}
    ft = pd.DataFrame(rows).T.reindex(columns=order, fill_value=0)
    ntot = {c: int((sub["cls"] == c).sum()) for c in order}
    print("\n=== features vs class (counts; events may have several) ===")
    print(ft.to_string())
    print("\nclass totals: " + ", ".join(f"{c}={n}" for c, n in ntot.items()))
    pct_f = ft.copy().astype(float)
    for c in order:
        pct_f[c] = 100 * ft[c] / ntot[c] if ntot[c] else np.nan
    print("\n=== features vs class (% of each class) ===")
    print(pct_f.round(1).to_string())

    if not plot:
        return
    for tab, name, ttl in ((pct_s, "shape", "primary shape"),
                           (pct_f, "features", "features")):
        fig, ax = plt.subplots(figsize=(1.6 * len(order) + 4, 0.5 * len(tab) + 3))
        im = ax.imshow(tab.values.astype(float), cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(tab.columns)), tab.columns)
        ax.set_yticks(range(len(tab.index)), tab.index)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=11,
                            color="white" if v > np.nanmax(tab.values) * 0.6 else "black")
        fig.colorbar(im, ax=ax).set_label("% of class")
        ax.set_title(f"{ttl} by class  [% of each class]")
        out = os.path.join(OUTDIR, f"morphstats_{name}.png")
        fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
        print(f"wrote {out}")
# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(
        description="General SE-SNe population plotter (scatter/hist/cdf/colorevol).")
    ap.add_argument("--kind", choices=["scatter", "scatter3", "contour", "hist",
                                       "cdf", "colorevol", "magevol", "colortrack", "morphstats"],)
    ap.add_argument("--c", help="third parameter -> marker colour (scatter3)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--x", help="parameter for x-axis (or the quantity for hist/cdf)")
    ap.add_argument("--y", help="parameter for y-axis (scatter only)")
    ap.add_argument("--type", nargs="+", default=None, help='type(s) or "all"')
    ap.add_argument("--limits", action="store_true")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--max-magerr", type=float, default=None)
    ap.add_argument("--list", action="store_true", help="list available parameters and exit")
    ap.add_argument("--alpha", type=float, default=0.12,
                    help="per-event track transparency (colortrack)")
    ap.add_argument("--separate", action="store_true",
                    help="colortrack: one figure per class")
    ap.add_argument("--axis", default="mag", choices=["mag", "days"],
                    help="colortrack x-axis: luminosity stage or phase in days")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.68, 0.95],
                    help="contour: fractions of events enclosed (e.g. 0.5 0.9)")
    ap.add_argument("--bw", type=float, default=None,
                    help="contour: KDE bandwidth (smaller = tighter to the points)")
    ap.add_argument("--normalise", choices=["event", "class"], default=None,
                    help="colortrack: remove the vertical offset (extinction test)")
    ap.add_argument("--ref-stage", type=float, default=0.0,
                    help="colortrack: stage used as the zero-point when normalising")
    ap.add_argument("--no-points", action="store_true")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="ZTFIDs (or substrings) to drop from the plot; "
                         "also reads fit/exclude_events.txt if present")
    ap.add_argument("--no-morph-cut", action="store_true",
                    help="colortrack: skip the below-1.5-mag / colorodd cut")
    ap.add_argument("--html", action="store_true",
                    help="interactive Plotly version (scatter/scatter3/contour/colortrack)")
    a = ap.parse_args()

    if a.list:
        print("Available parameters (--x / --y):")
        for k, v in PARAMS.items():
            tag = " [mag]" if v["is_mag"] else (" [has limits]" if v.get("lim") else "")
            print(f"  {k:22s} {v['label']}{tag}")
        return

    if not a.kind:
        ap.error("--kind is required (scatter/hist/cdf/colorevol), or use --list")

    df = _load()
    df = _apply_exclusions(df, extra=a.exclude)
    types = _types_arg(a.type)
    
    if a.kind == "morphstats":
        morph_stats(df, types)
        return

    if a.kind == "colorevol":
        for t in types:
            plot_colorevol(df, t, cmap_name=a.cmap)
        return
    
    if a.kind == "magevol":
        for t in types:
            plot_magevol(df, t, cmap_name=a.cmap)
        return
    
    if a.kind == "colortrack":
        if a.html:
            plot_colortrack_html(df, types, alpha=max(a.alpha, 0.15), axis=a.axis,
                                 normalise=a.normalise, ref_stage=a.ref_stage,
                                 morph_cut=not a.no_morph_cut)
        else:
            plot_colortrack(df, types, alpha=a.alpha, separate=a.separate,
                            axis=a.axis, normalise=a.normalise,
                            ref_stage=a.ref_stage, morph_cut=not a.no_morph_cut)
        return
    
    if not a.x:
        ap.error(f"--x is required for --kind {a.kind}")
    xp = _resolve(a.x)

    if a.kind == "scatter":
        if not a.y: ap.error("--y is required for scatter")
        (plot_scatter_html if a.html else plot_scatter)(
            df, xp, _resolve(a.y), types, limits=a.limits, max_magerr=a.max_magerr)
    elif a.kind == "scatter3":
        if not (a.y and a.c): ap.error("scatter3 needs --x, --y and --c")
        (plot_scatter3_html if a.html else plot_scatter3)(
            df, xp, _resolve(a.y), _resolve(a.c), types, cmap_name=a.cmap,
            vmin=a.vmin, vmax=a.vmax, max_magerr=a.max_magerr)
    elif a.kind == "contour":
        if not a.y: ap.error("--y is required for contour")
        (plot_contour_html if a.html else plot_scatter_contour)(
            df, xp, _resolve(a.y), types, fracs=a.levels, bw=a.bw,
            show_points=not a.no_points, max_magerr=a.max_magerr)
        
    elif a.kind == "hist":
        plot_hist(df, xp, types, bins=a.bins, density=a.density)
    elif a.kind == "cdf":
        plot_cdf(df, xp, types)


if __name__ == "__main__":
    main()