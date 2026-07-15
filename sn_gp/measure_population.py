"""
measure_population.py  --  turn the fitted light curves into one measurements
table for the population analysis.

For every event:
  1. drop it entirely if any remark contains 'epoch' or 'half_r'
  2. choose ONE fit by verdict priority good > keep > bad, BIC tie-break
  3. reconstruct that fit's GP mean (reconstruct.load_fit)
  4. for each colour g and r, pick the usable filter (ztf vs sdss) by sampling
     quality, then measure FWHM / rise / fade / peak on the GP mean
  5. peak absolute mag (per band), g-r colour (own peaks), peak-time separation
  6. rest-frame durations (divide by 1+z); abs mag via FlatLambdaCDM
  7. join SN type + z from BTS.csv

Writes fit/population_measurements.csv  (one row per event).

Run:
    python measure_population.py
    python measure_population.py --limit 20     # test on first 20 events
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

import config
import reconstruct

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

REVIEW  = os.path.join(config.FIT_ROOT, "fit_review.csv")
OUT     = os.path.join(config.FIT_ROOT, "population_measurements.csv")
BTS     = os.path.expanduser("~/GP_SN/BTS.csv")   # for type + redshift

# the four kernels, and how a group key maps to a kernel name
KERNELS = ["gibbs", "changepoint", "changepoint_1", "matern32"]
VERDICT_RANK = {"good": 0, "keep": 1, "bad": 2, "": 3}

# which filters realise each colour
G_FILTERS = ["ztfg", "sdssg"]
R_FILTERS = ["ztfr", "sdssr"]

F0_mJy = 3631e3   # AB zero point (same as elsewhere)

# --- band-selection thresholds (confirmed) ---
MIN_SIDE   = 2     # >=2 points each side of peak
BIN_DAYS   = 5.0   # bin width for coverage
MIN_BINS   = 3     # >=3 occupied bins over the span


# =====================================================================
# selection: best (event, kernel) fit
# =====================================================================
def load_review():
    d = pd.read_csv(REVIEW, dtype=str).fillna("")
    d["kernel"] = d["group"].str.replace("_constant", "", regex=False)
    return d

def load_bic():
    """(ZTFID, kernel) -> BIC, from the json records (authoritative)."""
    bic = {}
    for f in os.listdir(config.JSON_DIR):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(config.JSON_DIR, f)))
        bic[(r["name"], r["kernel"])] = float(r["BIC"])
    return bic

def choose_fits(review, bic):
    """Return {ZTFID: kernel} for the best fit per event, after dropping
       epoch/half_r events. Priority good>keep>bad, BIC tie-break."""
    chosen = {}
    for z, grp in review.groupby("ZTFID"):
        remarks = " ".join(grp["remark"].str.lower())
        if "epoch" in remarks or "half_r" in remarks:
            continue                                  # rule 1: drop entirely
        best = None
        for _, row in grp.iterrows():
            kern = row["kernel"]
            rank = VERDICT_RANK.get(row["verdict"].strip().lower(), 3)
            b = bic.get((z, kern), np.inf)
            key = (rank, b)
            if best is None or key < best[0]:
                best = (key, kern)
        if best is None:
            continue
        best_rank = best[0][0]                     # rank of the chosen fit
        if best_rank >= VERDICT_RANK["bad"]:       # all verdicts bad (or blank) -> drop
            continue
        if np.isfinite(bic.get((z, best[1]), np.inf)):
            chosen[z] = best[1]
    return chosen


# =====================================================================
# band sampling quality
# =====================================================================
def band_quality(phase):
    """Return (usable, n_pts, occupied_bins) for one band's phase array.
       usable = >=MIN_SIDE points each side of peak AND >=MIN_BINS occupied
       5-day bins across its span (rejects sparse and time-clustered bands)."""
    phase = np.asarray(phase, float)
    n = len(phase)
    if n == 0:
        return False, 0, 0
    n_before = int(np.sum(phase < -2))
    n_after  = int(np.sum(phase >  2))
    span = phase.max() - phase.min()
    if span <= 0:
        occ = 1
    else:
        edges = np.arange(phase.min(), phase.max() + BIN_DAYS, BIN_DAYS)
        occ = int(np.count_nonzero(np.histogram(phase, bins=edges)[0]))
    usable = (n_before >= MIN_SIDE) and (n_after >= MIN_SIDE) and (occ >= MIN_BINS)
    return usable, n, occ

def pick_filter(fit, filters):
    """Among candidate filters (e.g. ztfg/sdssg) present in this fit, pick the
       usable one with the most occupied bins (best time-distributed), tie by
       n_pts. Returns filter name or None."""
    best = None
    for b in filters:
        if b not in fit.obj["bands"]:
            continue
        ph, _, _ = fit.band_data(b)
        usable, n, occ = band_quality(ph)
        if not usable:
            continue
        key = (occ, n)                                # more bins, then more points
        if best is None or key > best[0]:
            best = (key, b)
    return best[1] if best else None


# =====================================================================
# FWHM / rise / fade on the GP mean
# =====================================================================
def measure_shape(fit, band, n=2000):
    """Measure on the GP mean within THIS band's own observed span (first to
       last detection), so a crossing is never found in an extrapolated region.
       If a side doesn't cross half-max within that span, that side is a lower
       limit (true crossing lies beyond the observed data), so fwhm_ll <= true FWHM."""
    ph, _, _ = fit.band_data(band)
    nan_out = dict(peak_flux=np.nan, peak_phase=np.nan, fwhm=np.nan,
                   rise=np.nan, fade=np.nan, fwhm_ll=np.nan, is_limit=False,
                   rise_measured=False, fade_measured=False)
    if ph.size == 0:
        return nan_out
    wave = config.WAVE_EFF_UM[band]
    tg = np.linspace(ph.min(), ph.max(), n)          # band's OWN data span
    _, mu, _ = fit.predict_grid(tg, wave)
    if not np.any(mu > 0):
        return nan_out
    ipk = int(np.argmax(mu)); Fpk = mu[ipk]; tpk = tg[ipk]; half = Fpk / 2.0

    tL = np.nan
    below = np.where(mu[:ipk + 1] < half)[0]
    if below.size:
        i0 = below[-1]
        if i0 + 1 <= ipk:
            f0, f1 = mu[i0], mu[i0 + 1]; t0, t1 = tg[i0], tg[i0 + 1]
            tL = t0 + (half - f0) * (t1 - t0) / (f1 - f0) if f1 != f0 else t0

    tR = np.nan
    below_r = np.where(mu[ipk:] < half)[0]
    if below_r.size:
        j = ipk + below_r[0]
        if j - 1 >= ipk:
            f0, f1 = mu[j - 1], mu[j]; t0, t1 = tg[j - 1], tg[j]
            tR = t0 + (half - f0) * (t1 - t0) / (f1 - f0) if f1 != f0 else t1

    rise = tpk - tL if np.isfinite(tL) else np.nan
    fade = tR - tpk if np.isfinite(tR) else np.nan
    fwhm = (rise + fade) if (np.isfinite(rise) and np.isfinite(fade)) else np.nan

    rise_measured = np.isfinite(rise)
    fade_measured = np.isfinite(fade)
    is_limit = not (rise_measured and fade_measured)
    r_part = rise if rise_measured else (tpk - tg[0])      # anchor to band data extent
    f_part = fade if fade_measured else (tg[-1] - tpk)
    fwhm_ll = r_part + f_part                              # true FWHM >= fwhm_ll

    return dict(peak_flux=float(Fpk), peak_phase=float(tpk),
                fwhm=fwhm, rise=rise, fade=fade,
                fwhm_ll=fwhm_ll, is_limit=is_limit,
                rise_measured=rise_measured, fade_measured=fade_measured)


def flux_to_absmag(flux_mJy, z):
    """Peak flux (mJy) -> absolute AB mag using luminosity distance from z.
       (No K-correction -- flagged in plots.)"""
    if not np.isfinite(flux_mJy) or flux_mJy <= 0 or not np.isfinite(z) or z <= 0:
        return np.nan
    m_app = -2.5 * np.log10(flux_mJy / F0_mJy)
    dL_pc = cosmo.luminosity_distance(z).to("pc").value
    return m_app - 5.0 * np.log10(dL_pc / 10.0)

def mag_at_phase(fit, band, phase, z):
    """Absolute AB mag of `band`'s GP mean at a specific observed phase.
       Used for fixed-epoch colours (e.g. both bands at the r-band peak time).
       Returns NaN if outside the band's observed span or flux<=0 there."""
    if band is None or not np.isfinite(phase):
        return np.nan
    ph, _, _ = fit.band_data(band)
    if ph.size == 0:
        return np.nan
    # only trust it within the band's own observed span (no extrapolation)
    if phase < ph.min() or phase > ph.max():
        return np.nan
    wave = config.WAVE_EFF_UM[band]
    _, mu, _ = fit.predict_grid(np.array([phase]), wave)
    flux = float(mu[0])
    return flux_to_absmag(flux, z)
# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N chosen events (testing)")
    args = ap.parse_args()

    review = load_review()
    bic = load_bic()
    chosen = choose_fits(review, bic)
    bts = pd.read_csv(BTS)
    zcol = "redshift" if "redshift" in bts.columns else ("z" if "z" in bts.columns else None)
    idcol = "ZTFID" if "ZTFID" in bts.columns else bts.columns[0]
    def _safe_z(v):
        try:
            z = float(v)
            return z if z > 0 else np.nan
        except (TypeError, ValueError):
            return np.nan                    # '-', '', NaN, or any non-numeric

    meta = {str(r[idcol]): (str(r.get("type", "")),
                            _safe_z(r[zcol]) if zcol else np.nan)
            for _, r in bts.iterrows()}

    items = sorted(chosen.items())
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} events to measure\n")

    rows = []
    for i, (z_id, kern) in enumerate(items, 1):
        sn_type, z = meta.get(z_id, ("", np.nan))
        try:
            fit = reconstruct.load_fit(
                os.path.join(config.JSON_DIR, f"{z_id}_{kern}_constant.json"))
        except Exception as e:
            print(f"[{i}] {z_id} {kern}: reconstruct FAILED ({e})")
            continue

        gband = pick_filter(fit, G_FILTERS)
        rband = pick_filter(fit, R_FILTERS)

        g = measure_shape(fit, gband) if gband else None
        r = measure_shape(fit, rband) if rband else None

        # per-band absolute peak mag
        Mg = flux_to_absmag(g["peak_flux"], z) if g else np.nan
        Mr = flux_to_absmag(r["peak_flux"], z) if r else np.nan

        # colour (each band's own peak); peak-time separation (g - r)
        color_gr = (Mg - Mr) if (np.isfinite(Mg) and np.isfinite(Mr)) else np.nan
        dt_peak = ((g["peak_phase"] - r["peak_phase"])
                   if (g and r and np.isfinite(g["peak_phase"]) and np.isfinite(r["peak_phase"]))
                   else np.nan)
        # peak epochs (observed phase) of each band
        tpk_g = g["peak_phase"] if g else np.nan
        tpk_r = r["peak_phase"] if r else np.nan

        # --- three colour definitions ---
        # (3) own-peak (existing): M_g,peak - M_r,peak
        color_ownpeak = (Mg - Mr) if (np.isfinite(Mg) and np.isfinite(Mr)) else np.nan

        # (1) at r-band peak epoch: m_g(t_r) - m_r(t_r)
        Mg_at_rpk = mag_at_phase(fit, gband, tpk_r, z)
        Mr_at_rpk = mag_at_phase(fit, rband, tpk_r, z)   # ~= Mr, but re-eval for consistency
        color_at_rpeak = ((Mg_at_rpk - Mr_at_rpk)
                          if (np.isfinite(Mg_at_rpk) and np.isfinite(Mr_at_rpk)) else np.nan)

        # (2) at g-band peak epoch: m_g(t_g) - m_r(t_g)
        Mg_at_gpk = mag_at_phase(fit, gband, tpk_g, z)   # ~= Mg
        Mr_at_gpk = mag_at_phase(fit, rband, tpk_g, z)
        color_at_gpeak = ((Mg_at_gpk - Mr_at_gpk)
                          if (np.isfinite(Mg_at_gpk) and np.isfinite(Mr_at_gpk)) else np.nan)
        # rest-frame durations
        zf = (1.0 + z) if np.isfinite(z) and z > 0 else np.nan
        def rest(x): return (x / zf) if (x is not None and np.isfinite(x) and np.isfinite(zf)) else np.nan

        # which band is intrinsically brightest (more negative abs mag)
        if np.isfinite(Mg) and np.isfinite(Mr):
            brightest = "g" if Mg < Mr else "r"
        elif np.isfinite(Mg):
            brightest = "g"
        elif np.isfinite(Mr):
            brightest = "r"
        else:
            brightest = ""

        rows.append(dict(
            ZTFID=z_id, type=sn_type, z=z, kernel=kern,
            g_filter=gband or "", r_filter=rband or "",
            # observed-frame shape
            fwhm_g_obs=g["fwhm"] if g else np.nan,
            fwhm_r_obs=r["fwhm"] if r else np.nan,
            rise_g_obs=g["rise"] if g else np.nan,
            rise_r_obs=r["rise"] if r else np.nan,
            fade_g_obs=g["fade"] if g else np.nan,
            fade_r_obs=r["fade"] if r else np.nan,
            # rest-frame shape
            fwhm_g=rest(g["fwhm"]) if g else np.nan,
            fwhm_r=rest(r["fwhm"]) if r else np.nan,
            rise_g=rest(g["rise"]) if g else np.nan,
            rise_r=rest(r["rise"]) if r else np.nan,
            fade_g=rest(g["fade"]) if g else np.nan,
            fade_r=rest(r["fade"]) if r else np.nan,
            # peaks / luminosity
            peakphase_g=g["peak_phase"] if g else np.nan,
            peakphase_r=r["peak_phase"] if r else np.nan,

            fwhm_g_ll=rest(g["fwhm_ll"]) if g else np.nan,
            fwhm_r_ll=rest(r["fwhm_ll"]) if r else np.nan,
            g_is_limit=bool(g["is_limit"]) if g else False,
            r_is_limit=bool(r["is_limit"]) if r else False,
            fwhm_brightest_ll=((rest(g["fwhm_ll"]) if g else np.nan) if brightest == "g"
                               else (rest(r["fwhm_ll"]) if r else np.nan) if brightest == "r"
                               else np.nan),
            brightest_is_limit=((bool(g["is_limit"]) if g else False) if brightest == "g"
                                else (bool(r["is_limit"]) if r else False) if brightest == "r"
                                else False),
            M_g=Mg, M_r=Mr, dt_peak_g_minus_r=dt_peak,
            brightest_band=brightest,
            M_brightest=(Mg if brightest == "g" else Mr if brightest == "r" else np.nan),
            fwhm_brightest=((rest(g["fwhm"]) if g else np.nan) if brightest == "g"
                            else (rest(r["fwhm"]) if r else np.nan) if brightest == "r"
                            else np.nan),
            color_gr=color_ownpeak,              # keep existing name = own-peak difference
            color_at_rpeak=color_at_rpeak,       # both bands at r-peak epoch
            color_at_gpeak=color_at_gpeak,       # both bands at g-peak epoch
        ))
        print(f"[{i}] {z_id:16s} {kern:13s} g={gband or '-':6s} r={rband or '-':6s} "
              f"FWHMr={rows[-1]['fwhm_r']:.1f}  Mr={Mr:.2f}" if np.isfinite(Mr)
              else f"[{i}] {z_id:16s} {kern:13s} g={gband or '-':6s} r={rband or '-':6s}  (r NaN)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} events")
    # quick coverage report
    for c in ("fwhm_g", "fwhm_r", "M_g", "M_r", "color_gr", "dt_peak_g_minus_r", "color_at_rpeak", "color_at_gpeak"):
        print(f"  {c:20s}: {int(df[c].notna().sum())}/{len(df)} finite")

    # lower-limit recovery (the payoff of the band-own-span change)
    for band in ("g", "r"):
        fin = int(df[f"fwhm_{band}"].notna().sum())
        lim = int((df[f"{band}_is_limit"].fillna(False) &
                   df[f"fwhm_{band}_ll"].notna()).sum())
        con = int((df[f"{band}_is_limit"].fillna(False) &
                   (df[f"fwhm_{band}_ll"] >= 16)).sum())
        print(f"  {band}: finite={fin}  lower-limits={lim}  "
              f"constraining(>=16d)={con}  plottable={fin + con}")


if __name__ == "__main__":
    main()
