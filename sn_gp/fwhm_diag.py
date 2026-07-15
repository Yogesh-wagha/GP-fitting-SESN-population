"""
fwhm_diagnostic.py  --  visualise how FWHM / rise / fade are measured for one
event, and report BOTH observed-frame and rest-frame durations.

It uses the SAME functions the analysis uses (measure_population.measure_shape
and .pick_filter), so what you see is exactly what went into the CSV. The plot
axis is in OBSERVED phase (so the shaded rise/fade regions line up with the
actual data points), but the printed box gives the rest-frame duration
(observed / (1+z)) as well -- and the rest-frame value matches the durlum plots.

By default it uses the fit the analysis chose for this event
(good>keep>bad, BIC tie-break); override with --kernel.

    python fwhm_diagnostic.py --name ZTF26aalsmpp
    python fwhm_diagnostic.py --name ZTF26aalsmpp --kernel matern32
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import reconstruct
import measure_population as mp     # reuse the exact measurement functions

BAND_COLOR = {"g": "green", "r": "red"}


def choose_kernel(name):
    """Same selection the analysis uses: good>keep>bad, BIC tie-break."""
    review = mp.load_review()
    bic = mp.load_bic()
    chosen = mp.choose_fits(review, bic)
    return chosen.get(name)


def lookup_z(ztfid):
    """Redshift from BTS.csv (same source/handling as measure_population)."""
    bts = pd.read_csv(mp.BTS)
    zcol = "redshift" if "redshift" in bts.columns else ("z" if "z" in bts.columns else None)
    idcol = "ZTFID" if "ZTFID" in bts.columns else bts.columns[0]
    if zcol is None:
        return np.nan
    row = bts[bts[idcol].astype(str) == ztfid]
    if row.empty:
        return np.nan
    try:
        z = float(row.iloc[0][zcol])
        return z if z > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def panel(ax, fit, band_letter, filters, z):
    band = mp.pick_filter(fit, filters)
    col = BAND_COLOR[band_letter]
    zf = (1.0 + z) if np.isfinite(z) and z > 0 else np.nan

    def rest(x):
        return (x / zf) if (np.isfinite(x) and np.isfinite(zf)) else np.nan

    if band is None:
        ax.text(0.5, 0.5, f"no usable {band_letter} band", ha="center",
                va="center", transform=ax.transAxes, color="grey")
        ax.set_title(f"{band_letter} band  (rejected / absent)")
        return

    # GP mean on a fine grid + data (OBSERVED phase on the axis)
    tg, mu, sd = fit.predict_band(band, n=2000)
    ph, fl, fe = fit.band_data(band)
    ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.15)
    ax.plot(tg, mu, color=col, lw=1.3, label=f"{band} GP mean")
    ax.errorbar(ph, fl, yerr=fe, fmt="o", ms=4, color=col, mfc="none",
                elinewidth=0.7, capsize=2, label="data")

    # the SAME measurement the analysis ran (observed-frame days)
    s = mp.measure_shape(fit, band)
    Fpk, tpk = s["peak_flux"], s["peak_phase"]
    fwhm, rise, fade = s["fwhm"], s["rise"], s["fade"]

    if not np.isfinite(Fpk):
        ax.set_title(f"{band}:  no positive peak")
        ax.legend(fontsize=7); return

    half = Fpk / 2.0
    ax.axhline(half, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(Fpk,  color=col, lw=0.6, ls=":", alpha=0.5)
    ax.axvline(tpk,  color="k", lw=0.8, ls="-", alpha=0.6)
    ax.annotate("half-max", xy=(tg.min(), half), xytext=(3, 3),
                textcoords="offset points", fontsize=7, va="bottom")

    zlab = f", z={z:.3f}" if np.isfinite(z) else ", z=?"
    txt = [f"peak @ {tpk:.1f} d (obs)"]

    # rise side
    if np.isfinite(rise):
        tL = tpk - rise
        ax.plot([tL], [half], "kv", ms=7)
        ax.axvspan(tL, tpk, color=col, alpha=0.08)
        ax.annotate(f"rise = {rise:.1f} d", xy=((tL + tpk) / 2, half),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8)
        txt.append(f"rise = {rise:.1f} obs / {rest(rise):.1f} rest")
    else:
        txt.append("rise = n/a (no left crossing)")

    # fade side
    if np.isfinite(fade):
        tR = tpk + fade
        ax.plot([tR], [half], "k^", ms=7)
        ax.axvspan(tpk, tR, color=col, alpha=0.16)
        ax.annotate(f"fade = {fade:.1f} d", xy=((tpk + tR) / 2, half),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8)
        txt.append(f"fade = {fade:.1f} obs / {rest(fade):.1f} rest")
    else:
        txt.append("fade = n/a (no right crossing)")

    # FWHM: both frames; rest-frame is what the durlum plots use
    if np.isfinite(fwhm):
        txt.append(f"FWHM = {fwhm:.1f} obs / {rest(fwhm):.1f} rest{zlab}")
    else:
        # one-sided: show the lower limit in both frames
        ll = s.get("fwhm_ll", np.nan)
        if np.isfinite(ll):
            txt.append(f"FWHM >= {ll:.1f} obs / {rest(ll):.1f} rest (lower limit){zlab}")
        else:
            txt.append("FWHM = n/a")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_title(f"{band}")
    ax.set_xlabel("observed phase [days]"); ax.set_ylabel("flux [mJy]")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", lw=0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kernel", default=None,
                    help="override; default = the fit the analysis chose")
    ap.add_argument("--mean", default="constant")
    a = ap.parse_args()

    kern = a.kernel or choose_kernel(a.name)
    if kern is None:
        raise SystemExit(f"{a.name}: no chosen fit found (dropped, or not in review)")
    path = os.path.join(config.JSON_DIR, f"{a.name}_{kern}_{a.mean}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    fit = reconstruct.load_fit(path)
    z = lookup_z(a.name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    panel(axes[0], fit, "g", mp.G_FILTERS, z)
    panel(axes[1], fit, "r", mp.R_FILTERS, z)
    zt = f"z={z:.3f}" if np.isfinite(z) else "z unknown"
    fig.suptitle(f"{a.name}   kernel={kern}   {zt}   "
                 f"(axis: observed phase; box gives rest-frame too)", fontsize=11)
    out = f"fwhm_{a.name}_{kern}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    if np.isfinite(z):
        print(f"  z={z:.4f}  (rest-frame durations = observed / {1+z:.4f})")
    else:
        print("  z unknown -> rest-frame durations shown as n/a")


if __name__ == "__main__":
    main()
