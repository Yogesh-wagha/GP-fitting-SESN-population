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

# --- rest-frame K-correction ---
LAMBDA_G_REST_UM = 0.470     # rest-frame g effective wavelength (matches ztfg/sdssg)
KCORR_BW_SIGN = +1.0         # bandwidth term: M = m_shifted - DM + SIGN*2.5*log10(1+z)
                             # VALIDATE vs BTS peakabs; set -1.0 if residual grows with z


COLOR_SCAN_PHASES = np.arange(-10.0, 10.0001, 2.5)   # rest-frame days vs r-peak
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
    _, mu,sd = fit.predict_grid(tg, wave)
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
                rise_measured=rise_measured, fade_measured=fade_measured, peak_flux_err=float(sd[ipk]),)


def flux_to_absmag(flux_mJy, z):
    """Peak flux (mJy) -> absolute AB mag using luminosity distance from z.
       (No K-correction -- flagged in plots.)"""
    if not np.isfinite(flux_mJy) or flux_mJy <= 0 or not np.isfinite(z) or z <= 0:
        return np.nan
    m_app = -2.5 * np.log10(flux_mJy / F0_mJy)
    dL_pc = cosmo.luminosity_distance(z).to("pc").value
    return m_app - 5.0 * np.log10(dL_pc / 10.0)

def restframe_g_absmag(fit, z, gband, rband):
    """Rest-frame g-band peak absolute mag via GP evaluated at observed
       wavelength lambda_g*(1+z) (SED cross-band shift), plus the (1+z)
       bandwidth K-term. Returns (M, sigma_M, peak_phase, wave_extrap)."""
    nanout = (np.nan, np.nan, np.nan, False)
    if not (np.isfinite(z) and z > 0):
        return nanout
    lam_obs = LAMBDA_G_REST_UM * (1.0 + z)          # samples rest-frame g
    spans = []
    for b in (gband, rband):
        if b is None:
            continue
        ph, _, _ = fit.band_data(b)
        if ph.size:
            spans.append((ph.min(), ph.max()))
    if not spans:
        return nanout
    t0 = min(s[0] for s in spans); t1 = max(s[1] for s in spans)
    LAMBDA_R_UM = 0.617
    wave_extrap = lam_obs > LAMBDA_R_UM + 1e-6
    tg = np.linspace(t0, t1, 2000)
    _, mu, sd = fit.predict_grid(tg, lam_obs)
    if not np.any(mu > 0):
        return nanout
    ipk = int(np.argmax(mu))
    fpk, spk, tpk = float(mu[ipk]), float(sd[ipk]), float(tg[ipk])
    m_shifted = -2.5 * np.log10(fpk / F0_mJy)
    dL_pc = cosmo.luminosity_distance(z).to("pc").value
    DM = 5.0 * np.log10(dL_pc / 10.0)
    M = m_shifted - DM + KCORR_BW_SIGN * 2.5 * np.log10(1.0 + z)
    sigma_M = (2.5 / np.log(10.0)) * (spk / fpk)    # GP flux sd -> mag error
    return M, sigma_M, tpk, wave_extrap


def absmag_err_at_phase(fit, band, phase, z):
    """Observed-frame absolute AB mag and 1-sigma error of `band` GP mean at
       observed `phase`. For colours (DM cancels in the difference, but we keep
       it so the number is a magnitude). NaN if outside the band's span."""
    if band is None or not np.isfinite(phase):
        return np.nan, np.nan
    ph, _, _ = fit.band_data(band)
    if ph.size == 0 or phase < ph.min() or phase > ph.max():
        return np.nan, np.nan
    wave = config.WAVE_EFF_UM[band]
    _, mu, sd = fit.predict_grid(np.array([phase]), wave)
    f, s = float(mu[0]), float(sd[0])
    if f <= 0 or not (np.isfinite(z) and z > 0):
        return np.nan, np.nan
    dL_pc = cosmo.luminosity_distance(z).to("pc").value
    DM = 5.0 * np.log10(dL_pc / 10.0)
    M = -2.5 * np.log10(f / F0_mJy) - DM            # observed-frame (no K) for colours
    sigma_M = (2.5 / np.log(10.0)) * (s / f)
    return M, sigma_M

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

        # per-band peak-mag 1-sigma (propagate GP flux sd at peak)
        Mg_err = ((2.5/np.log(10)) * g["peak_flux_err"]/g["peak_flux"]
                  if (g and np.isfinite(g["peak_flux"]) and g["peak_flux"] > 0) else np.nan)
        Mr_err = ((2.5/np.log(10)) * r["peak_flux_err"]/r["peak_flux"]
                  if (r and np.isfinite(r["peak_flux"]) and r["peak_flux"] > 0) else np.nan)

        # rest-frame g absolute mag (K-corrected) + error + extrapolation flag
        M_rest_g, M_rest_g_err, _, kcorr_extrap = restframe_g_absmag(fit, z, gband, rband)

        # ---- colours (observed frame) with 1-sigma errors ----
        # own-peak
        color_ownpeak = (Mg - Mr) if (np.isfinite(Mg) and np.isfinite(Mr)) else np.nan
        color_ownpeak_err = (np.hypot(Mg_err, Mr_err)
                             if np.isfinite(Mg_err) and np.isfinite(Mr_err) else np.nan)
        # at r-peak epoch
        mg_rp, mg_rp_e = absmag_err_at_phase(fit, gband, tpk_r, z)
        mr_rp, mr_rp_e = absmag_err_at_phase(fit, rband, tpk_r, z)
        color_rpeak = (mg_rp - mr_rp) if (np.isfinite(mg_rp) and np.isfinite(mr_rp)) else np.nan
        color_rpeak_err = (np.hypot(mg_rp_e, mr_rp_e)
                           if np.isfinite(mg_rp_e) and np.isfinite(mr_rp_e) else np.nan)
        # at g-peak epoch
        mg_gp, mg_gp_e = absmag_err_at_phase(fit, gband, tpk_g, z)
        mr_gp, mr_gp_e = absmag_err_at_phase(fit, rband, tpk_g, z)
        color_gpeak = (mg_gp - mr_gp) if (np.isfinite(mg_gp) and np.isfinite(mr_gp)) else np.nan
        color_gpeak_err = (np.hypot(mg_gp_e, mr_gp_e)
                           if np.isfinite(mg_gp_e) and np.isfinite(mr_gp_e) else np.nan)
        # +10 rest-frame days after r-peak  (observed offset = 10*(1+z))
        zf = (1.0 + z) if np.isfinite(z) and z > 0 else np.nan
        t_10 = (tpk_r + 10.0 * zf) if (np.isfinite(tpk_r) and np.isfinite(zf)) else np.nan
        mg_10, mg_10_e = absmag_err_at_phase(fit, gband, t_10, z)
        mr_10, mr_10_e = absmag_err_at_phase(fit, rband, t_10, z)
        color_10d = (mg_10 - mr_10) if (np.isfinite(mg_10) and np.isfinite(mr_10)) else np.nan
        color_10d_err = (np.hypot(mg_10_e, mr_10_e)
                         if np.isfinite(mg_10_e) and np.isfinite(mr_10_e) else np.nan)
        
        
        
        # rest-frame durations
        zf = (1.0 + z) if np.isfinite(z) and z > 0 else np.nan

        # colour scanned from -10 to +10 rest-frame days about r-peak (2.5-d steps)
        color_scan = {}
        for p in COLOR_SCAN_PHASES:
            t_obs = (tpk_r + p * zf) if (np.isfinite(tpk_r) and np.isfinite(zf)) else np.nan
            mg_p, _ = absmag_err_at_phase(fit, gband, t_obs, z)
            mr_p, _ = absmag_err_at_phase(fit, rband, t_obs, z)
            color_scan[f"color_p{p:+.1f}"] = ((mg_p - mr_p)
                if (np.isfinite(mg_p) and np.isfinite(mr_p)) else np.nan)

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
            # per-band peak-mag errors
            M_g_err=Mg_err, M_r_err=Mr_err,
            # rest-frame g luminosity (K-corrected)
            M_rest_g=M_rest_g, M_rest_g_err=M_rest_g_err, kcorr_wave_extrap=bool(kcorr_extrap),
            # colours + errors (observed frame)
            color_gr=color_ownpeak, color_gr_err=color_ownpeak_err,
            color_at_rpeak=color_rpeak, color_at_rpeak_err=color_rpeak_err,
            color_at_gpeak=color_gpeak, color_at_gpeak_err=color_gpeak_err,
            color_10d=color_10d, color_10d_err=color_10d_err,
            **color_scan,
            
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
    if "M_rest_g" in df.columns:
        n_rest = int(df["M_rest_g"].notna().sum())
        n_extrap = int(df["kcorr_wave_extrap"].fillna(False).sum())
        n_extrap_finite = int((df["kcorr_wave_extrap"].fillna(False) &
                               df["M_rest_g"].notna()).sum())
        print(f"  M_rest_g            : {n_rest}/{len(df)} finite  "
              f"({n_extrap} wave-extrap, {n_extrap_finite} finite)")
    if "color_10d" in df.columns:
        print(f"  color_10d           : {int(df['color_10d'].notna().sum())}/{len(df)} finite")


if __name__ == "__main__":
    main()
