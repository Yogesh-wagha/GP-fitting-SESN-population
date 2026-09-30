"""
dashboard.py  --  browse every analysis-ready SE-SN light curve on one page,
filter by class / primary shape / features, and inspect individual events.

MAIN PAGE
  * sticky, collapsible filter bar: class buttons (+ All), primary-shape
    buttons, feature buttons (match any / all), candidates toggle, sort menu
  * scrollable two-column grid of GP fits in FLUX space (g/r/i), lazy-loaded

EVENT PANEL (click any card)
  * GP fit in flux AND magnitude space
  * every measured parameter
  * duration-luminosity plot with the event as a yellow star
  * colour tracks of the event's class with the event highlighted
  * histogram of any parameter (drop-down) with the event marked + percentile
  * flag the event as a candidate odd-one-out (saved to CSV)

"Analysis-ready" = present in population_measurements.csv and not tagged
very_bad in morphology_review.csv.

All images live in a NEW directory, fit/dashboard/, and are cached:
  thumbs/ and per-event light curves are keyed on the fit JSON's modification
  time (refits regenerate automatically); population plots (dur-lum, colour
  tracks, histograms) are keyed on the CSVs' modification times.

    python dashboard.py                    # http://localhost:5004
    python dashboard.py --precache         # render all thumbnails, then exit
    python dashboard.py --precache-detail  # thumbnails + per-event panels
    python dashboard.py --clear-cache
    ssh -L 5004:localhost:5004 ariywagh@prospero.ljmu.ac.uk
"""

import os
import io
import csv
import shutil
import argparse
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure          # thread-safe OO API, no pyplot state
from astropy.cosmology import FlatLambdaCDM
from flask import (Flask, request, send_file, render_template_string,
                   abort, jsonify)

import config
import reconstruct

PORT = 5004
MEAS = os.path.join(config.FIT_ROOT, "population_measurements.csv")
MORPH = os.path.join(config.FIT_ROOT, "morphology_review.csv")

DASH = os.path.join(config.FIT_ROOT, "dashboard")          # new directory for this work
THUMB_DIR = os.path.join(DASH, "thumbs")
DETAIL_DIR = os.path.join(DASH, "detail")
CAND_CSV = os.path.join(DASH, "candidates.csv")
for _d in (DASH, THUMB_DIR, DETAIL_DIR):
    os.makedirs(_d, exist_ok=True)

F0_mJy = 3631e3
REF_DROP_MAG = 1.5
BAND_COLOR = {"g": "#1b9e2f", "r": "#d62728", "i": "#8c564b"}
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)          # same cosmology as measure_population


def distmod(z):
    """Distance modulus from redshift; NaN if z is missing/invalid."""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return np.nan
    if not (np.isfinite(z) and z > 0):
        return np.nan
    return float(5.0 * np.log10(cosmo.luminosity_distance(z).to("pc").value / 10.0))

TYPE_TO_CLASS = {"SN Ic": "Ic", "SN Ib": "Ib", "SN Ib/c": "Ib/c",
                 "SN Ic-BL": "Ic-BL", "SN Ic-BL?": "Ic-BL", "SLSN-I": "SLSN-I"}
CLASS_ORDER = ["Ic", "Ib", "Ib/c", "Ic-BL", "SLSN-I"]
CLASS_COLOR = {"Ic": "#3300FF", "Ib": "#FFAA00", "Ib/c": "#E100FF",
               "Ic-BL": "#FF0000", "SLSN-I": "#00C853"}
CLASS_MARKER = {"Ic": "o", "Ib": "^", "Ib/c": "D", "Ic-BL": "s", "SLSN-I": "*"}

# same vocabulary as morph_review.py
SHAPES = [
    ("single",    "Single smooth peak (normal)"),
    ("double",    "Double peak"),
    ("earlybump", "Early bump then main peak"),
    ("plateau",   "Plateau / flat top"),
    ("linear",    "Linear decline"),
    ("irregular", "Irregular / no clean shape"),
    ("unclear",   "Too sparse to judge"),
]
FEATURES = [
    ("bumpy",       "Bumpy / undulating"),
    ("shoulder",    "Shoulder or knee on decline"),
    ("rebright",    "Late re-brightening"),
    ("sharp",       "Sharp / narrow peak"),
    ("broad",       "Broad / rounded peak"),
    ("fastdecline", "Fast decline"),
    ("slowtail",    "Slow, shallow tail"),
    ("asymmetric",  "Strongly asymmetric"),
    ("colorodd",    "g and r shapes disagree"),
    ("noisy",       "Noisy / poor sampling"),
]
MORPH_COLS = ["shape", "features", "below15", "very_bad", "notes"]

MAIN_SHOW = [
    ("z", "redshift"), ("kernel", "kernel"),
    ("g_filter", "g filter"), ("r_filter", "r filter"),
    ("fwhm_r", "FWHM r, rest [d]"), ("fwhm_g", "FWHM g, rest [d]"),
    ("rise_r", "rise r, rest [d]"), ("fade_r", "fade r, rest [d]"),
    ("rise_frac_r", "rise / FWHM (r)"),
    ("fwhm_r_ll", "FWHM r lower limit [d]"), ("r_is_limit", "r FWHM is a limit"),
    ("M_rest_g", "M rest-g (K-corr)"), ("M_rest_g_err", "M rest-g error"),
    ("M_rest_g_mw", "M rest-g (K + MW)"),
    ("M_r", "M_r (obs)"), ("M_g", "M_g (obs)"),
    ("color_at_rpeak", "g-r at r-peak"), ("color_10d", "g-r at +10 d"),
    ("color_gr", "g-r own peaks"),
    ("dt_peak_g_minus_r", "dt_gr [d]"),
    ("dm15_r", "dm15 r"), ("dm15_g", "dm15 g"),
    ("tail_slope_r", "tail slope r [mag/d]"),
    ("tail_slope_r_npts", "pts in tail window"),
    ("fwhm_ratio_gr", "FWHM ratio g/r"), ("color_rate", "d(g-r)/dt [mag/d]"),
    ("dcolor_pre", "d(g-r) pre-peak"), ("dcolor_post", "d(g-r) post-peak"),
    ("dcolor_full", "d(g-r) full"),
    ("kcorr_wave_extrap", "K-corr wavelength extrapolated"),
]
HIST_MAIN = ["fwhm_r", "fwhm_g", "rise_r", "fade_r", "rise_frac_r", "M_rest_g",
             "M_r", "M_g", "color_at_rpeak", "color_10d", "color_gr",
             "dt_peak_g_minus_r", "dm15_r", "tail_slope_r", "fwhm_ratio_gr",
             "color_rate", "dcolor_pre", "dcolor_post", "dcolor_full", "z"]
NON_PARAM = {"ZTFID", "type", "cls", "kernel", "g_filter", "r_filter",
             "brightest_band"} | set(MORPH_COLS)
SCAN_PREFIXES = ("color_m", "color_p")

LOCK = threading.Lock()        # serialises GP reconstruction (TF/gpflow)
POP_LOCK = threading.Lock()    # population plots (no TF; separate so they don't wait)


# ======================================================================= data
_DF = {"key": None, "df": None}

def _mtime(p):
    return int(os.path.getmtime(p)) if os.path.exists(p) else 0

def pop_key():
    return f"{_mtime(MEAS)}_{_mtime(MORPH)}"

def load_df():
    """Analysis-ready events: measured, and not tagged very_bad."""
    key = pop_key()
    if _DF["key"] == key:
        return _DF["df"]
    df = pd.read_csv(MEAS)
    if os.path.exists(MORPH):
        m = pd.read_csv(MORPH, dtype=str).fillna("")
        keep = ["ZTFID"] + [c for c in MORPH_COLS if c in m.columns]
        df = df.merge(m[keep].drop_duplicates("ZTFID"), on="ZTFID", how="left")
    for c in MORPH_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    df = df[df["very_bad"].str.lower() != "yes"].copy()
    df["cls"] = df["type"].map(TYPE_TO_CLASS).fillna(df["type"].astype(str))
    df = df.reset_index(drop=True)
    _DF.update(key=key, df=df)
    return df

def get_row(zid):
    df = load_df()
    m = df[df["ZTFID"] == zid]
    if m.empty:
        abort(404)
    return df, m.iloc[0]

def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan

def numcol(df, c):
    if c not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[c], errors="coerce")

def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v if v.strip() else "--"
    f = num(v)
    if not np.isfinite(f):
        return "--"
    if float(f).is_integer() and abs(f) < 1e6:
        return f"{int(f)}"
    return f"{f:.4g}"

def _feat_list(s):
    return [f for f in str(s).split(";") if f]


# ======================================================================= fits
def json_path(zid, kernel):
    return os.path.join(config.JSON_DIR, f"{zid}_{kernel}_constant.json")

@lru_cache(maxsize=24)
def _fit_cached(path, mtime):
    return reconstruct.load_fit(path)

def get_fit(zid, kernel):
    p = json_path(zid, kernel)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no fit JSON: {os.path.basename(p)}")
    return _fit_cached(p, _mtime(p))

def _letter(b):
    n = str(b).split("::")[-1]
    for p in ("sdss", "ztf", "atlas", "ps1"):
        if n.startswith(p):
            n = n[len(p):]
    return n[:1].lower() if n else "?"

def _flux_to_mag(f):
    f = np.asarray(f, float)
    out = np.full_like(f, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok] / F0_mJy)
    return out

def _gri(fit, gf, rf):
    allb = list(fit.obj["bands"])
    sel = [b for b in (gf, rf) if isinstance(b, str) and b in allb]
    sel += [b for b in allb if _letter(b) == "i" and b not in sel]
    if not sel:
        sel = [b for b in allb if _letter(b) in ("g", "r", "i")]
    return sel

def _rband(fit, rf, bands):
    if isinstance(rf, str) and rf in fit.obj["bands"]:
        return rf
    return next((b for b in bands if _letter(b) == "r"), None)

def _r_peak(fit, rb):
    """(peak_mag, peak_phase, peak_flux) of the r-band GP mean."""
    if rb is None:
        return np.nan, np.nan, np.nan
    tg, mu, _ = fit.predict_band(rb, n=1200)
    m = _flux_to_mag(mu)
    if not np.any(np.isfinite(m)):
        return np.nan, np.nan, np.nan
    i = int(np.nanargmin(m))
    return float(m[i]), float(tg[i]), float(mu[i])

def _stage_cols(df, prefix):
    found = []
    for c in df.columns:
        if c.startswith(prefix):
            try:
                found.append((float(c[len(prefix):]), c))
            except ValueError:
                pass
    if not found:
        raise KeyError(f"no {prefix}* columns in the CSV")
    found.sort()
    return np.array([f[0] for f in found]), [f[1] for f in found]


# ==================================================================== drawing
def _draw_flux(ax, fit, bands, small=False, ref_flux=np.nan, tpk=np.nan):
    fs = 8 if small else 11
    for b in bands:
        col = BAND_COLOR.get(_letter(b), "grey")
        tg, mu, sd = fit.predict_band(b, n=400)
        ax.plot(tg, mu, color=col, lw=1.4, label=str(b))
        ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ph, fl, fe = fit.band_data(b)
        ax.errorbar(ph, fl, yerr=fe, fmt="o", ms=3 if small else 4, color=col,
                    mfc="white", mec=col, elinewidth=0.7,
                    capsize=0 if small else 1.5, zorder=3)
    if np.isfinite(ref_flux):
        ax.axhline(ref_flux, color="k", ls="--", lw=1.1, alpha=0.7)
    if np.isfinite(tpk):
        ax.axvline(tpk, color="k", ls=":", lw=0.9, alpha=0.5)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("phase [d]", fontsize=fs)
    ax.set_ylabel("flux [mJy]", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.legend(fontsize=fs - 1, loc="best", frameon=False)
    ax.grid(alpha=0.2)

def _draw_mag(ax, fit, bands, ref_mag=np.nan, tpk=np.nan, dm=np.nan, small=False):
    """Magnitude-space panel. Left axis = apparent mag; if the distance modulus
       `dm` is finite, a right-hand axis shows absolute mag = m - DM (a pure
       shift: observed frame, no K-correction, no extinction)."""
    fs = 8 if small else 11
    yv = []
    for b in bands:
        col = BAND_COLOR.get(_letter(b), "grey")
        tg, mu, sd = fit.predict_band(b, n=400 if small else 500)
        m = _flux_to_mag(mu)
        lo, hi = _flux_to_mag(mu + sd), _flux_to_mag(mu - sd)
        ok = np.isfinite(m)
        if ok.any():
            ax.plot(tg[ok], m[ok], color=col, lw=1.4, label=str(b))
            yv.append(m[ok])
            bo = ok & np.isfinite(lo) & np.isfinite(hi)
            if bo.any():
                ax.fill_between(tg[bo], lo[bo], hi[bo], color=col, alpha=0.18, lw=0)
        ph, fl, fe = fit.band_data(b)
        mm = _flux_to_mag(fl)
        with np.errstate(invalid="ignore", divide="ignore"):
            me = (2.5 / np.log(10)) * np.abs(np.asarray(fe, float) /
                                             np.asarray(fl, float))
        k = np.isfinite(mm)
        if k.any():
            ax.errorbar(np.asarray(ph)[k], mm[k], yerr=me[k], fmt="o",
                        ms=3 if small else 4, color=col, mfc="white", mec=col,
                        elinewidth=0.7, capsize=0 if small else 1.5, zorder=3)
            yv.append(mm[k])
    if np.isfinite(ref_mag):
        ax.axhline(ref_mag, color="k", ls="--", lw=1.0 if small else 1.1, alpha=0.7)
        if not small:
            ax.text(0.01, ref_mag, f" r-peak +{REF_DROP_MAG} mag", va="bottom",
                    ha="left", fontsize=9, transform=ax.get_yaxis_transform())
        yv.append(np.array([ref_mag]))
    if np.isfinite(tpk):
        ax.axvline(tpk, color="k", ls=":", lw=0.9, alpha=0.5)
    if yv:
        v = np.concatenate(yv)
        v = v[np.isfinite(v)]
        if v.size:
            lo_, hi_ = float(v.min()), float(v.max())
            pad = 0.08 * ((hi_ - lo_) or 1.0)
            ax.set_ylim(hi_ + pad, lo_ - pad)              # inverted: bright at top
    ax.set_xlabel("phase [d]", fontsize=fs)
    ax.set_ylabel("apparent mag (AB)", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.legend(fontsize=fs - 1, loc="best", frameon=False)
    ax.grid(alpha=0.2)
    if np.isfinite(dm):
        # linear shift, so the right axis inverts together with the left one
        sec = ax.secondary_yaxis("right", functions=(lambda m, d=dm: m - d,
                                                     lambda M, d=dm: M + d))
        sec.set_ylabel("absolute mag (m - DM)" if small
                       else "absolute mag = m - DM  (no K-corr.)", fontsize=fs)
        sec.tick_params(labelsize=fs - 1)

def _pop_path(name):
    d = os.path.join(DETAIL_DIR, f"pop_{pop_key()}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ==================================================================== renders
def render_thumb(row, space="flux"):
    space = "mag" if space == "mag" else "flux"
    zid, kern = row["ZTFID"], str(row["kernel"])
    jp = json_path(zid, kern)
    if not os.path.exists(jp):
        raise FileNotFoundError("no fit JSON")
    tag = "" if space == "flux" else "_mag"     # flux keeps its old name -> old cache valid
    out = os.path.join(THUMB_DIR, f"{zid}_{kern}{tag}_{_mtime(jp)}.png")
    if os.path.exists(out):
        return out
    with LOCK:
        if not os.path.exists(out):
            fit = get_fit(zid, kern)
            bands = _gri(fit, row.get("g_filter"), row.get("r_filter"))
            fig = Figure(figsize=(5.6, 3.1), dpi=90)
            ax = fig.subplots()
            if space == "flux":
                _draw_flux(ax, fit, bands, small=True)
            else:
                rb = _rband(fit, row.get("r_filter"), bands)
                mpk, tpk, _ = _r_peak(fit, rb)
                ref = mpk + REF_DROP_MAG if np.isfinite(mpk) else np.nan
                _draw_mag(ax, fit, bands, ref_mag=ref, tpk=tpk,
                          dm=distmod(row.get("z")), small=True)
            fig.tight_layout()
            fig.savefig(out)
    return out

def render_lc(row):
    zid, kern = row["ZTFID"], str(row["kernel"])
    jp = json_path(zid, kern)
    out = os.path.join(DETAIL_DIR, f"lc2_{zid}_{kern}_{_mtime(jp)}.png")   # lc2: has abs-mag axis
    if os.path.exists(out):
        return out
    with LOCK:
        if not os.path.exists(out):
            fit = get_fit(zid, kern)
            bands = _gri(fit, row.get("g_filter"), row.get("r_filter"))
            rb = _rband(fit, row.get("r_filter"), bands)
            mpk, tpk, fpk = _r_peak(fit, rb)
            ref_mag = mpk + REF_DROP_MAG if np.isfinite(mpk) else np.nan
            ref_flux = fpk * 10 ** (-REF_DROP_MAG / 2.5) if np.isfinite(fpk) else np.nan
            fig = Figure(figsize=(14, 4.8), dpi=100)
            ax1, ax2 = fig.subplots(1, 2)
            _draw_flux(ax1, fit, bands, ref_flux=ref_flux, tpk=tpk)
            ax1.set_title("flux space", fontsize=12)
            _draw_mag(ax2, fit, bands, ref_mag=ref_mag, tpk=tpk,
                      dm=distmod(row.get("z")))
            ax2.set_title("magnitude space", fontsize=12)
            fig.tight_layout()
            fig.savefig(out)
    return out

def render_durlum(df, row):
    out = _pop_path(f"durlum_{row['ZTFID']}.png")
    if os.path.exists(out):
        return out
    with POP_LOCK:
        if os.path.exists(out):
            return out
        fig = Figure(figsize=(7.2, 5.6), dpi=100)
        ax = fig.subplots()
        x, y = numcol(df, "fwhm_r"), numcol(df, "M_rest_g")
        for c in CLASS_ORDER:
            m = (df["cls"] == c) & x.notna() & y.notna()
            if m.any():
                ax.scatter(x[m], y[m], s=24, marker=CLASS_MARKER[c],
                           color=CLASS_COLOR[c], alpha=0.45, edgecolor="none",
                           label=f"{c} ({int(m.sum())})")
        xe, ye = num(row.get("fwhm_r")), num(row.get("M_rest_g"))
        tag = ""
        if not np.isfinite(xe) and np.isfinite(num(row.get("fwhm_r_ll"))):
            xe, tag = num(row.get("fwhm_r_ll")), "  (FWHM lower limit)"
        if np.isfinite(xe) and np.isfinite(ye):
            ax.scatter([xe], [ye], marker="*", s=560, color="yellow",
                       edgecolor="black", linewidth=1.4, zorder=6,
                       label=row["ZTFID"] + tag)
            if tag:
                ax.annotate("", xy=(xe + 10, ye), xytext=(xe, ye), zorder=7,
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
        else:
            ax.text(0.5, 0.03, "no finite FWHM_r / M_rest_g for this event",
                    transform=ax.transAxes, ha="center", color="crimson")
        ax.invert_yaxis()
        ax.set_xlabel("rest-frame FWHM r [d]")
        ax.set_ylabel("M rest-g (K-corrected)")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out)
    return out

def render_colortrack(df, row, norm="raw"):
    norm = norm if norm in ("raw", "event") else "raw"
    out = _pop_path(f"ct_{row['ZTFID']}_{norm}.png")
    if os.path.exists(out):
        return out
    with POP_LOCK:
        if os.path.exists(out):
            return out
        levels, cols = _stage_cols(df, "color_m")
        cls = row["cls"]
        col = CLASS_COLOR.get(cls, "grey")
        sub = df[df["cls"] == cls]
        # background = the same cut used for the analysis colour tracks
        cut = ((sub["below15"].str.lower() == "yes") &
               ~sub["features"].map(lambda s: "colorodd" in _feat_list(s)))
        vals = sub[cut][cols].apply(pd.to_numeric, errors="coerce")
        ev = pd.to_numeric(row[cols], errors="coerce").values.astype(float)
        iref = int(np.argmin(np.abs(levels)))
        note = ""
        if norm == "event":
            vals = vals.sub(vals.iloc[:, iref], axis=0)
            if np.isfinite(ev[iref]):
                ev = ev - ev[iref]
            else:
                ev = np.full_like(ev, np.nan)
                note = "  (no colour at peak: cannot normalise)"

        fig = Figure(figsize=(7.2, 5.6), dpi=100)
        ax = fig.subplots()
        n = 0
        for _, r_ in vals.iterrows():
            yy = r_.values.astype(float)
            ok = np.isfinite(yy)
            if ok.sum() >= 2:
                ax.plot(levels[ok], yy[ok], color=col, lw=0.9, alpha=0.15, zorder=1)
                n += 1
        if n:
            med = vals.median(axis=0, skipna=True).values.astype(float)
            ax.plot(levels, med, color=col, lw=3, zorder=3,
                    label=f"{cls} median (n={n})")
        ok = np.isfinite(ev)
        if ok.any():
            ax.plot(levels[ok], ev[ok], color="black", lw=3.4, zorder=5)
            ax.plot(levels[ok], ev[ok], color="yellow", lw=1.6, marker="o", ms=4,
                    mec="black", mew=0.5, zorder=6, label=row["ZTFID"])
        passes = bool(cut.get(row.name, False))
        ax.axvline(0, color="k", ls=":", lw=1, alpha=0.6)
        if norm == "event":
            ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel("r-band stage [mag from peak]  (-ve rise, +ve decline)")
        ax.set_ylabel("g-r [mag]" if norm == "raw" else "g-r minus value at peak [mag]")
        ax.set_title(f"{row['ZTFID']} {'passes' if passes else 'FAILS'} "
                     f"the colour-track cut{note}", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out)
    return out

def render_hist(df, row, param):
    if param not in df.columns:
        raise KeyError(f"unknown parameter '{param}'")
    safe = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in param)
    out = _pop_path(f"hist_{row['ZTFID']}_{safe}.png")
    if os.path.exists(out):
        return out
    with POP_LOCK:
        if os.path.exists(out):
            return out
        v = numcol(df, param)
        ve = num(row.get(param))
        allv = v.dropna()
        fig = Figure(figsize=(8, 4.6), dpi=100)
        ax = fig.subplots()
        if allv.empty:
            ax.text(0.5, 0.5, f"no finite values of {param}", ha="center",
                    transform=ax.transAxes)
        else:
            edges = np.histogram_bin_edges(allv.values, bins=30)
            ax.hist(allv.values, bins=edges, color="0.85", label=f"all ({len(allv)})")
            for c in CLASS_ORDER:
                d = v[df["cls"] == c].dropna()
                if len(d):
                    ax.hist(d.values, bins=edges, histtype="step", lw=1.8,
                            color=CLASS_COLOR[c], label=f"{c} ({len(d)})")
            if np.isfinite(ve):
                ax.axvline(ve, color="black", lw=3)
                ax.axvline(ve, color="yellow", lw=1.5)
                top = ax.get_ylim()[1]
                ax.plot([ve], [top * 0.93], marker="*", ms=22, color="yellow",
                        mec="black", mew=1.2, zorder=6)
                dc = v[df["cls"] == row["cls"]].dropna()
                pc = 100 * (dc < ve).mean() if len(dc) else np.nan
                pa = 100 * (allv < ve).mean()
                ax.set_title(f"{param} = {ve:.4g}   |   below this value: "
                             f"{pc:.0f}% of {row['cls']}, {pa:.0f}% of all",
                             fontsize=11)
            else:
                ax.set_title(f"{param}: not measured for {row['ZTFID']}",
                             fontsize=11, color="crimson")
            if param.startswith("M_"):
                ax.invert_xaxis()
            ax.legend(fontsize=9, loc="best")
        ax.set_xlabel(param)
        ax.set_ylabel("number of events")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(out)
    return out


# ================================================================= candidates
def load_cands():
    if not os.path.exists(CAND_CSV):
        return {}
    d = pd.read_csv(CAND_CSV, dtype=str).fillna("")
    return dict(zip(d["ZTFID"], d["note"]))

def save_cand(zid, flag, note):
    c = load_cands()
    if flag:
        c[zid] = note
    else:
        c.pop(zid, None)
    with open(CAND_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ZTFID", "note"])
        for k in sorted(c):
            w.writerow([k, c[k]])


def error_png(msg):
    fig = Figure(figsize=(6, 2.2), dpi=90)
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="crimson", wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ================================================================== templates
INDEX = """
<!doctype html><html><head><meta charset="utf-8"><title>SE-SNe dashboard</title>
<style>
 body{margin:0;font-family:system-ui,Arial,sans-serif;background:#f4f4f4}
 #bar{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #ccc;
      box-shadow:0 2px 4px rgba(0,0,0,.06);padding:6px 12px;max-height:38vh;overflow-y:auto}
 #bar.collapsed .body{display:none}
 .head{display:flex;align-items:center;gap:12px;font-size:14px;flex-wrap:wrap}
 .row{margin:5px 0;display:flex;flex-wrap:wrap;align-items:center;gap:4px}
 .lab{font-size:12px;color:#666;width:64px;flex-shrink:0}
 .btn{border:1px solid #bbb;background:#fafafa;border-radius:12px;padding:2px 9px;
      font-size:12px;cursor:pointer;user-select:none}
 .btn.on{background:#1e3a5f;color:#fff;border-color:#1e3a5f}
 .btn small{opacity:.7}
 .tog{cursor:pointer;border:1px solid #bbb;border-radius:4px;padding:1px 8px;
      background:#eee;font-size:12px;user-select:none}
 #grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px}
 .card{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden;cursor:pointer}
 .card:hover{border-color:#1e3a5f;box-shadow:0 1px 6px rgba(0,0,0,.15)}
 .card .t{display:flex;gap:8px;align-items:center;padding:4px 8px;font-size:13px;
          border-bottom:1px solid #eee}
 .card .t .id{font-weight:700}
 .pill{border-radius:8px;padding:0 6px;font-size:11px;color:#fff}
 .sh{background:#eee;border-radius:8px;padding:0 6px;font-size:11px}
 .ft{font-size:11px;color:#777;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
 .cand{color:#b8860b;font-weight:700;display:none}
 .card.is-cand .cand{display:inline}
 .card img{width:100%;display:block;min-height:120px;background:#fafafa}
 #modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:50}
 #modal .box{position:absolute;top:3vh;bottom:3vh;left:4vw;right:4vw;background:#fff;
             border-radius:8px;overflow:hidden}
 #modal iframe{width:100%;height:100%;border:0}
 #modal .x{position:absolute;top:6px;right:10px;z-index:2;font-size:18px;border:1px solid #ccc;
           background:#fff;cursor:pointer;border-radius:50%;width:32px;height:32px}
</style></head><body>

<div id="bar">
  <div class="head">
    <span class="tog" onclick="toggleBar()">&#9776; filters</span>
    <b>SE-SNe light curves</b>
    <span>showing <b id="count">{{ n }}</b> / {{ n }}</span>
    <span id="summary" style="color:#666;font-size:12px"></span>
    <span style="margin-left:auto;font-size:12px">view
      <span class="btn on" id="sp-flux" onclick="setSpace('flux')">flux</span>
      <span class="btn" id="sp-mag" onclick="setSpace('mag')">mag</span></span>
    <span style="font-size:12px">sort
      <select id="sortsel" onchange="sortCards()">
        <option value="id">ZTF name</option>
        <option value="z">redshift</option>
        <option value="m">M rest-g</option>
        <option value="fwhm">FWHM r</option>
        <option value="col">g-r at r-peak</option>
      </select></span>
  </div>
  <div class="body">
    <div class="row"><span class="lab">class</span>
      <span class="btn on" id="cls-all" onclick="setAll()">All <small>{{ n }}</small></span>
      {% for c, k in classes %}
      <span class="btn" data-g="cls" data-v="{{ c }}" onclick="tog(this)">{{ c }} <small>{{ k }}</small></span>
      {% endfor %}
    </div>
    <div class="row"><span class="lab">shape</span>
      {% for key, lab, k in shapes %}
      <span class="btn" data-g="shape" data-v="{{ key }}" title="{{ lab }}" onclick="tog(this)">{{ key }} <small>{{ k }}</small></span>
      {% endfor %}
      <span class="tog" onclick="clearG('shape')">clear</span>
    </div>
    <div class="row"><span class="lab">features</span>
      {% for key, lab, k in features %}
      <span class="btn" data-g="feat" data-v="{{ key }}" title="{{ lab }}" onclick="tog(this)">{{ key }} <small>{{ k }}</small></span>
      {% endfor %}
      <span class="tog" id="fmode" onclick="toggleMode()">match: any</span>
      <span class="tog" onclick="clearG('feat')">clear</span>
      <span class="btn" data-g="cand" data-v="1" onclick="tog(this)">&#9733; candidates <small id="ncand">{{ ncand }}</small></span>
    </div>
  </div>
</div>

<div id="grid">
{% for r in cards %}
  <div class="card{% if r.cand %} is-cand{% endif %}" data-id="{{ r.id }}" data-cls="{{ r.cls }}"
       data-shape="{{ r.shape }}" data-feat="{{ r.feat }}" data-cand="{{ 1 if r.cand else 0 }}"
       data-z="{{ r.z }}" data-m="{{ r.m }}" data-fwhm="{{ r.fwhm }}" data-col="{{ r.col }}"
       onclick="openEv('{{ r.id }}')">
    <div class="t"><span class="id">{{ r.id }}</span>
      <span class="pill" style="background:{{ r.ccol }}">{{ r.cls }}</span>
      <span class="sh">{{ r.shape or 'untagged' }}</span>
      <span class="cand">&#9733;</span>
      <span class="ft">{{ r.feat_disp }}</span></div>
    <img loading="lazy" src="/thumb/{{ r.id }}" alt="{{ r.id }}">
  </div>
{% endfor %}
</div>

<div id="modal"><div class="box">
  <button class="x" onclick="closeEv()">&#10005;</button>
  <iframe id="mframe"></iframe></div></div>

<script>
const S = {cls:new Set(), shape:new Set(), feat:new Set(), cand:new Set(), mode:'any'};
const cards = Array.from(document.querySelectorAll('.card'));
function tog(el){
  const g = el.dataset.g, v = el.dataset.v;
  if (S[g].has(v)) { S[g].delete(v); el.classList.remove('on'); }
  else { S[g].add(v); el.classList.add('on'); }
  if (g === 'cls') document.getElementById('cls-all').classList.toggle('on', S.cls.size === 0);
  apply();
}
function setAll(){
  S.cls.clear();
  document.querySelectorAll('[data-g="cls"]').forEach(b => b.classList.remove('on'));
  document.getElementById('cls-all').classList.add('on');
  apply();
}
function clearG(g){
  S[g].clear();
  document.querySelectorAll('[data-g="' + g + '"]').forEach(b => b.classList.remove('on'));
  apply();
}
function toggleMode(){
  S.mode = (S.mode === 'any') ? 'all' : 'any';
  document.getElementById('fmode').textContent = 'match: ' + S.mode;
  apply();
}
function toggleBar(){ document.getElementById('bar').classList.toggle('collapsed'); }
function setSpace(sp){
  document.getElementById('sp-flux').classList.toggle('on', sp === 'flux');
  document.getElementById('sp-mag').classList.toggle('on', sp === 'mag');
  // lazy images stay lazy: only cards near the viewport fetch the new version
  for (const c of cards)
    c.querySelector('img').src = '/thumb/' + c.dataset.id + '?space=' + sp;
}
function apply(){
  let n = 0;
  for (const c of cards){
    const shp = c.dataset.shape || 'untagged';
    const f = (c.dataset.feat || '').split(';').filter(Boolean);
    let ok = (S.cls.size === 0 || S.cls.has(c.dataset.cls)) &&
             (S.shape.size === 0 || S.shape.has(shp)) &&
             (S.cand.size === 0 || c.dataset.cand === '1');
    if (ok && S.feat.size){
      const h = [...S.feat].map(x => f.includes(x));
      ok = (S.mode === 'any') ? h.some(Boolean) : h.every(Boolean);
    }
    c.style.display = ok ? '' : 'none';
    if (ok) n++;
  }
  document.getElementById('count').textContent = n;
  const parts = [];
  if (S.cls.size) parts.push([...S.cls].join(' + '));
  if (S.shape.size) parts.push('shape: ' + [...S.shape].join(' | '));
  if (S.feat.size) parts.push('features (' + S.mode + '): ' + [...S.feat].join(', '));
  if (S.cand.size) parts.push('candidates only');
  document.getElementById('summary').textContent = parts.join('   \u00b7   ');
}
function sortCards(){
  const k = document.getElementById('sortsel').value, g = document.getElementById('grid');
  const key = c => (k === 'id') ? c.dataset.id : parseFloat(c.dataset[k]);
  cards.sort((a, b) => {
    const x = key(a), y = key(b);
    if (k === 'id') return x < y ? -1 : (x > y ? 1 : 0);
    const nx = isNaN(x), ny = isNaN(y);
    if (nx && ny) return 0; if (nx) return 1; if (ny) return -1;
    return x - y;
  });
  cards.forEach(c => g.appendChild(c));
}
function openEv(id){
  document.getElementById('mframe').src = '/detail/' + id;
  document.getElementById('modal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}
function closeEv(){
  document.getElementById('modal').style.display = 'none';
  document.getElementById('mframe').src = 'about:blank';
  document.body.style.overflow = '';
}
function markCand(id, on){
  const c = cards.find(x => x.dataset.id === id);
  if (!c) return;
  c.dataset.cand = on ? '1' : '0';
  c.classList.toggle('is-cand', on);
  document.getElementById('ncand').textContent = cards.filter(x => x.dataset.cand === '1').length;
  apply();
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeEv(); });
document.getElementById('modal').addEventListener('click', e => { if (e.target.id === 'modal') closeEv(); });
</script>
</body></html>
"""

DETAIL = """
<!doctype html><html><head><meta charset="utf-8"><title>{{ zid }}</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;padding:14px 18px;background:#fff}
 h2{margin:0 40px 4px 0;font-size:20px}
 .meta{font-size:13px;color:#444;margin-bottom:8px}
 .pill{border-radius:8px;padding:1px 7px;font-size:12px;color:#fff}
 .sec{margin:16px 0 6px;font-weight:700;font-size:15px;border-bottom:1px solid #ddd;padding-bottom:3px}
 img{max-width:100%;border:1px solid #eee}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:1px 18px}
 .kv{display:flex;justify-content:space-between;border-bottom:1px solid #f2f2f2;font-size:13px;padding:2px 0}
 .kv .k{color:#666} .kv .v{font-weight:600}
 .two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .ctl{font-size:13px;margin:4px 0}
 .tb{border:1px solid #bbb;border-radius:10px;padding:2px 10px;cursor:pointer;background:#fafafa;font-size:12px}
 .tb.on{background:#1e3a5f;color:#fff}
 details summary{cursor:pointer;font-size:13px;color:#1e3a5f}
 .flag{background:#fff8dc;border:1px solid #e0c060;border-radius:6px;padding:8px;margin:14px 0;font-size:13px}
 a{color:#1e3a5f}
</style></head><body>

<h2>{{ zid }} <span class="pill" style="background:{{ ccol }}">{{ cls }}</span></h2>
<div class="meta">z = {{ z }} &middot; kernel = {{ kernel }} &middot; shape = <b>{{ shape or 'untagged' }}</b>
  &middot; features = {{ feat or '--' }}{% if notes %} &middot; notes: <i>{{ notes }}</i>{% endif %}
  &middot; <a href="https://fritz.science/source/{{ zid }}" target="_blank">Fritz &#8599;</a></div>

<div class="flag"><b>&#9733; Candidate odd-one-out?</b>
  <label><input type="checkbox" id="cflag" {% if is_cand %}checked{% endif %}> flag</label>
  <input type="text" id="cnote" value="{{ cand_note }}" placeholder="why is it odd?" style="width:55%">
  <button onclick="saveCand()">save</button> <span id="cmsg" style="color:#2e7d32"></span></div>

<div class="sec">GP fit (g/r/i)</div>
<img src="/img/lc/{{ zid }}">

<div class="sec">Measured parameters</div>
<div class="grid">{% for k, v in main %}<div class="kv"><span class="k">{{ k }}</span><span class="v">{{ v }}</span></div>{% endfor %}</div>
<details style="margin-top:6px"><summary>other columns ({{ other|length }})</summary>
  <div class="grid">{% for k, v in other %}<div class="kv"><span class="k">{{ k }}</span><span class="v">{{ v }}</span></div>{% endfor %}</div></details>
<details style="margin-top:4px"><summary>colour scans ({{ scans|length }})</summary>
  <div class="grid">{% for k, v in scans %}<div class="kv"><span class="k">{{ k }}</span><span class="v">{{ v }}</span></div>{% endfor %}</div></details>

<div class="two">
  <div><div class="sec">Duration&ndash;luminosity</div>
    <img src="/img/durlum/{{ zid }}"></div>
  <div><div class="sec">{{ cls }} colour tracks</div>
    <div class="ctl"><span class="tb on" id="n-raw" onclick="ct('raw')">raw</span>
      <span class="tb" id="n-event" onclick="ct('event')">event-normalised</span></div>
    <img id="ctimg" src="/img/ct/{{ zid }}?norm=raw"></div>
</div>

<div class="sec">Where does it sit? Histogram of a parameter</div>
<div class="ctl">parameter
  <select id="psel" onchange="hist()">
    <optgroup label="main">{% for p in hmain %}<option value="{{ p }}" {% if p == hdefault %}selected{% endif %}>{{ p }}</option>{% endfor %}</optgroup>
    <optgroup label="other">{% for p in hother %}<option value="{{ p }}">{{ p }}</option>{% endfor %}</optgroup>
  </select></div>
<img id="himg" src="/img/hist/{{ zid }}/{{ hdefault }}">

<script>
function ct(n){
  document.getElementById('ctimg').src = '/img/ct/{{ zid }}?norm=' + n;
  document.getElementById('n-raw').classList.toggle('on', n === 'raw');
  document.getElementById('n-event').classList.toggle('on', n === 'event');
}
function hist(){
  const p = document.getElementById('psel').value;
  document.getElementById('himg').src = '/img/hist/{{ zid }}/' + encodeURIComponent(p);
}
function saveCand(){
  const on = document.getElementById('cflag').checked;
  const note = document.getElementById('cnote').value;
  fetch('/flag/{{ zid }}', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({flag: on, note: note})})
    .then(r => r.json()).then(j => {
      document.getElementById('cmsg').textContent = 'saved';
      if (window.parent && window.parent.markCand) window.parent.markCand('{{ zid }}', on);
    });
}
</script></body></html>
"""


# ===================================================================== routes
app = Flask(__name__)

def _dattr(v):
    f = num(v)
    return "" if not np.isfinite(f) else f"{f:.5g}"

@app.route("/")
def index():
    df = load_df()
    cands = load_cands()
    cards = []
    for _, r in df.iterrows():
        feat = str(r["features"])
        cards.append(dict(
            id=r["ZTFID"], cls=r["cls"], shape=str(r["shape"]),
            feat=feat, feat_disp=feat.replace(";", " \u00b7 "),
            ccol=CLASS_COLOR.get(r["cls"], "#777"), cand=r["ZTFID"] in cands,
            z=_dattr(r.get("z")), m=_dattr(r.get("M_rest_g")),
            fwhm=_dattr(r.get("fwhm_r")), col=_dattr(r.get("color_at_rpeak"))))
    classes = [(c, int((df["cls"] == c).sum())) for c in CLASS_ORDER
               if (df["cls"] == c).any()]
    classes += [(c, int((df["cls"] == c).sum()))
                for c in sorted(set(df["cls"]) - set(CLASS_ORDER))]
    shp = df["shape"].replace("", "untagged")
    shapes = [(k, lab, int((shp == k).sum()))
              for k, lab in SHAPES + [("untagged", "no morphology tag")]
              if (shp == k).any()]
    fl = df["features"].map(_feat_list)
    features = [(k, lab, int(fl.map(lambda L, k=k: k in L).sum())) for k, lab in FEATURES]
    return render_template_string(INDEX, cards=cards, n=len(df), classes=classes,
                                  shapes=shapes, features=features,
                                  ncand=sum(1 for z in df["ZTFID"] if z in cands))

@app.route("/thumb/<zid>")
def thumb(zid):
    _, row = get_row(zid)
    try:
        return send_file(render_thumb(row, request.args.get("space", "flux")),
                         mimetype="image/png")
    except Exception as e:
        return error_png(f"{zid}: {e}")

@app.route("/detail/<zid>")
def detail(zid):
    df, row = get_row(zid)
    cands = load_cands()
    main, seen = [], set()
    for c, lab in MAIN_SHOW:
        if c in row.index:
            main.append((lab, fmt(row[c])))
            seen.add(c)
    scans = [(c, fmt(row[c])) for c in row.index if c.startswith(SCAN_PREFIXES)]
    other = [(c, fmt(row[c])) for c in row.index
             if c not in seen and c not in NON_PARAM and not c.startswith(SCAN_PREFIXES)]
    hmain = [p for p in HIST_MAIN if p in df.columns]
    hother = [c for c in df.columns
              if c not in NON_PARAM and c not in hmain
              and pd.api.types.is_numeric_dtype(df[c])
              and not pd.api.types.is_bool_dtype(df[c])]
    hdefault = "fwhm_r" if "fwhm_r" in hmain else (hmain + hother)[0]
    return render_template_string(
        DETAIL, zid=zid, cls=row["cls"], ccol=CLASS_COLOR.get(row["cls"], "#777"),
        z=fmt(row.get("z")), kernel=row.get("kernel"), shape=row["shape"],
        feat=str(row["features"]).replace(";", ", "), notes=row["notes"],
        main=main, other=other, scans=scans,
        hmain=hmain, hother=hother, hdefault=hdefault,
        is_cand=zid in cands, cand_note=cands.get(zid, ""))

@app.route("/img/lc/<zid>")
def img_lc(zid):
    _, row = get_row(zid)
    try:
        return send_file(render_lc(row), mimetype="image/png")
    except Exception as e:
        return error_png(f"light curve failed: {e}")

@app.route("/img/durlum/<zid>")
def img_durlum(zid):
    df, row = get_row(zid)
    try:
        return send_file(render_durlum(df, row), mimetype="image/png")
    except Exception as e:
        return error_png(f"dur-lum failed: {e}")

@app.route("/img/ct/<zid>")
def img_ct(zid):
    df, row = get_row(zid)
    try:
        return send_file(render_colortrack(df, row, request.args.get("norm", "raw")),
                         mimetype="image/png")
    except Exception as e:
        return error_png(f"colour track failed: {e}")

@app.route("/img/hist/<zid>/<param>")
def img_hist(zid, param):
    df, row = get_row(zid)
    try:
        return send_file(render_hist(df, row, param), mimetype="image/png")
    except Exception as e:
        return error_png(f"histogram failed: {e}")

@app.route("/flag/<zid>", methods=["POST"])
def flag(zid):
    get_row(zid)
    d = request.get_json(force=True, silent=True) or {}
    save_cand(zid, bool(d.get("flag")), str(d.get("note", "")))
    return jsonify(ok=True)


# ======================================================================== cli
def precache(detail=False):
    df = load_df()
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        try:
            render_thumb(r, "flux")
            render_thumb(r, "mag")          # same reconstructed fit, reused from the LRU cache
            if detail:
                render_lc(r)
                render_durlum(df, r)
                render_colortrack(df, r, "raw")
            print(f"[{i}/{n}] {r['ZTFID']}", flush=True)
        except Exception as e:
            print(f"[{i}/{n}] {r['ZTFID']} FAILED: {e}", flush=True)

def clear_cache():
    for d in (THUMB_DIR, DETAIL_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    print(f"cleared {THUMB_DIR} and {DETAIL_DIR} (candidates.csv kept)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--precache", action="store_true",
                    help="render every thumbnail, then exit")
    ap.add_argument("--precache-detail", action="store_true",
                    help="thumbnails + per-event light curves/dur-lum/colour tracks")
    ap.add_argument("--clear-cache", action="store_true")
    a = ap.parse_args()
    if a.clear_cache:
        clear_cache()
    if a.precache or a.precache_detail:
        precache(detail=a.precache_detail)
    elif not a.clear_cache:
        print(f"dashboard -> http://localhost:{a.port}")
        app.run(host="0.0.0.0", port=a.port, threaded=True, debug=False)