"""
compare_to_bts.py  --  validate GP-derived duration & luminosity against the
BTS Sample Explorer values (Perley linear-interpolation: peakabs, duration).

Per event, plots mine (y) vs BTS (x) with the 1:1 line, plus a residual panel.
This is a CONSISTENCY check, not a 1:1 expectation -- known offsets exist:
  * BTS peakabs has K-correction (2.5*log10(1+z)) AND Galactic extinction (A_V);
    your M has neither -> your M is fainter (less negative) by ~K + A_V.
    Use --apply-corr to add the SAME K-corr + A_V to your values so the
    luminosity comparison is like-for-like.
  * BTS duration is the hybrid two-band linear-interpolation timescale; yours is
    single-band FWHM on the GP mean. Expect correlation with scatter.
  * Confirm BTS 'duration' frame (obs vs rest) first; --bts-dur-frame sets it.

    python compare.py --band r
    python compare.py --band brightest --apply-corr
    python compare.py --band r --bts-dur-frame obs
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
    "legend.fontsize":  12,   # legend text
    "legend.title_fontsize": 13,
})

import config
import measure_population as mp

MINE = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "population_plots")
os.makedirs(OUTDIR, exist_ok=True)

# same coarse-group colours you use elsewhere (edit to taste)
TYPE_COLOR = {
    "SN Ic-BL": "#FF0000", "SN Ic-BL?": "#FF0000",
    "SN Ic": "#3300FF", "SN Ib": "#3300FF", "SN Ib/c": "#3300FF",
    "SLSN-I": "#00FF44",
}
def _col(t): return TYPE_COLOR.get(str(t), "grey")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_joined(band):
    mine = pd.read_csv(MINE)
    bts = pd.read_csv(mp.BTS)
    # BTS columns of interest
    keep = ["ZTFID", "peakabs", "duration", "A_V", "redshift"]
    keep = [c for c in keep if c in bts.columns]
    b = bts[keep].copy()
    for c in ("peakabs", "duration", "A_V", "redshift"):
        if c in b.columns:
            b[c] = _num(b[c])

    if band == "brightest":
        mine_M, mine_fwhm = "M_brightest", "fwhm_brightest"
    else:
        mine_M, mine_fwhm = f"M_{band}", f"fwhm_{band}"
    m = mine[["ZTFID", "type", "z", mine_M, mine_fwhm]].copy()
    m = m.rename(columns={mine_M: "M_mine", mine_fwhm: "fwhm_mine"})

    j = m.merge(b, on="ZTFID", how="inner")
    return j


def _panel_pair(x, y, types, xlabel, ylabel, title, out, invert=False):
    """Scatter y vs x with 1:1 line + residual (y - x) panel below."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y, types = x[m], y[m], types[m]
    if len(x) == 0:
        print(f"no overlapping finite points for {title}; skipping"); return

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.5, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))

    cols = [_col(t) for t in types]
    ax.scatter(x, y, c=cols, s=40, edgecolor="k", linewidth=0.4, alpha=0.85)

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    line = [lo - pad, hi + pad]
    ax.plot(line, line, "k--", lw=1, alpha=0.7, label="1:1")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.2)

    resid = y - x
    axr.scatter(x, resid, c=cols, s=30, edgecolor="k", linewidth=0.3, alpha=0.85)
    axr.axhline(0, color="k", lw=1, ls="--", alpha=0.7)
    med = np.median(resid)
    axr.axhline(med, color="crimson", lw=1, ls="-", alpha=0.7,
                label=f"median = {med:+.2f}")
    axr.set_xlabel(xlabel); axr.set_ylabel("GP - Linear")
    axr.legend(fontsize=10, loc="best")
    axr.grid(alpha=0.2)

    if invert:
        ax.invert_xaxis(); ax.invert_yaxis()

    # simple stats in the corner
    r = np.corrcoef(x, y)[0, 1]
    scatter = np.std(resid)
    ax.text(0.02, 0.98,
            f"n = {len(x)}\nr = {r:.3f}\nmedian off = {med:+.2f}\nrms = {scatter:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", lw=0.6))

    fig.tight_layout()
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out}  (n={len(x)}, r={r:.3f}, median offset={med:+.2f}, rms={scatter:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="r",
                    choices=["g", "r", "brightest"])
    ap.add_argument("--apply-corr", action="store_true",
                    help="add K-correction (2.5log10(1+z)) + A_V to YOUR M so the "
                         "luminosity comparison matches BTS conventions")
    ap.add_argument("--bts-dur-frame", default="rest", choices=["rest", "obs"],
                    help="frame of BTS 'duration' column (confirm first!). If 'obs', "
                         "it is divided by (1+z) to compare with your rest-frame FWHM.")
    a = ap.parse_args()

    j = load_joined(a.band)

    # ---- luminosity comparison ----
    M_mine = j["M_mine"].astype(float).copy()
    if a.apply_corr:
        z = j["z"].astype(float)
        kcorr = 2.5 * np.log10(1.0 + z)          # Perley's uniform K-correction
        av = j["A_V"].astype(float) if "A_V" in j.columns else 0.0
        # extinction makes intrinsic brighter (more negative); K-corr likewise per Perley sign
        M_mine = M_mine - kcorr - av.fillna(0.0)
        corr_tag = "_corr"
        ylab = "GP peak abs mag (K-corrected)"
    else:
        corr_tag = ""
        ylab = "GP peak abs mag (no K, no A_V)"

    if "peakabs" in j.columns:
        _panel_pair(
            x=j["peakabs"].astype(float).values, y=M_mine.values,
            types=j["type"].values,
            xlabel="Linear peakabs", ylabel=ylab,
            title=f"Luminosity: GP vs Linear  ({a.band} band)",
            out=os.path.join(OUTDIR, f"compare_lum_{a.band}{corr_tag}.png"),
            invert=True)   # magnitudes: brighter (more -ve) toward top/right

    # ---- duration comparison ----
    if "duration" in j.columns:
        bts_dur = j["duration"].astype(float).copy()
        if a.bts_dur_frame == "obs":
            zf = 1.0 + j["z"].astype(float)
            bts_dur = bts_dur / zf               # bring BTS obs duration to rest-frame
            dur_note = " (BTS dur /(1+z))"
        else:
            dur_note = ""
        _panel_pair(
            x=bts_dur.values, y=j["fwhm_mine"].astype(float).values,
            types=j["type"].values,
            xlabel=f"BTS duration [d]{dur_note}",
            ylabel="GP rest-frame FWHM [d]",
            title=f"Duration: GP vs Linear  ({a.band} band)",
            out=os.path.join(OUTDIR, f"compare_dur_{a.band}.png"),
            invert=False)


if __name__ == "__main__":
    main()
