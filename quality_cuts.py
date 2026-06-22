"""
quality_cuts.py  --  build clean light curves by removing bad detections.

Pipeline per object (all sample-independent, no per-object day cutoffs):
  1. pre-cuts        : drop alert_fp, dedupe, error floor + upper-error cut
  2. main body       : split on adaptive time gaps, keep the brightest segment
  3. robust clip     : stiff GP per band + iterative sigma-clipping
  4. peak from model : argmax of the GP mean (not a single point) + cross-band check
  5. active window   : keep only the contiguous region where the model is detected
  6. coverage gate   : flag objects with too few epochs / no bracketed peak

Reads : <BASE>/BTS_csv/*.csv          (raw)
Writes: <BASE>/BTS_csv_clean/*.csv    (cleaned, same columns, fewer rows)
        <BASE>/quality_report.csv     (one row per object; NOT review_flags.csv)

To re-review: point CSV_DIR in plot_detections.py at BTS_csv_clean, delete the
cached PNGs in review_site/plots/, and run review_app.py. Your review_flags.csv
remarks are untouched and will reload as before.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

# ---- paths ----
BASE      = "/home/yogesh1729/myWork/air_phd/gaussian_process_BTS"
CSV_DIR   = os.path.join(BASE, "BTS_csv")
CLEAN_DIR = os.path.join(BASE, "BTS_csv_clean")
REPORT    = os.path.join(BASE, "quality_report.csv")

G_FILTERS = ["ztfg", "sdssg"]
R_FILTERS = ["ztfr", "sdssr"]
GR = G_FILTERS + R_FILTERS

# ---- tunable knobs (uniform across the whole sample) ----
F0_mJy        = 3631e3      # AB zero-point
MAX_MAGERR    = 0.30        # upper error cut (absolute)
MAD_K         = 5.0         # also cut errors > median + MAD_K * MAD
ERR_FLOOR     = 0.03        # systematic flux-error floor (3%)
GAP_FACTOR    = 10.0        # a gap is "big" if > GAP_FACTOR * median cadence ...
GAP_MIN_DAYS  = 20.0        #   ... but never split on gaps below this
MIN_SEG_PTS   = 3           # a segment must have >= this many points to be "the SN"
LS_BOUNDS     = (5.0, 80.0) # GP length-scale bounds (days) -> stiff, no 1-pt spikes
CLIP_SIGMA    = 4.0         # reject points > this many sigma from the model
CLIP_MAX_ITER = 5
WIN_FLUX_FRAC = 0.01        # window edge where model drops below 1% of peak ...
WIN_NOISE_K   = 1.5         #   ... or 1.5 * median error, whichever is higher
MIN_EPOCHS    = 5           # coverage gate: per reference band
MIN_COLOR     = 4           # coverage gate: per band, for colour work


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def mag_to_flux(mag, magerr):
    flux = F0_mJy * 10.0 ** (-mag / 2.5)
    fluxerr = flux * (np.log(10.0) / 2.5) * magerr
    fluxerr = np.sqrt(fluxerr ** 2 + (ERR_FLOOR * flux) ** 2)   # systematic floor
    return flux, fluxerr


def pre_cuts(df):
    """Drop alert_fp, dedupe epochs, cut anomalous errors."""
    df = df[df["origin"] != "alert_fp"].copy()
    df = df[df["filter"].isin(GR)]

    # dedupe: same filter + rounded mjd -> keep highest SNR
    df = (df.sort_values("snr", ascending=False)
            .assign(_m=df["mjd"].round(2))
            .drop_duplicates(["filter", "_m"]).drop(columns="_m"))

    # upper error cut: absolute, plus robust (median + k*MAD)
    me = df["magerr"].to_numpy()
    med = np.median(me)
    mad = np.median(np.abs(me - med)) * 1.4826 + 1e-9
    keep = (me <= MAX_MAGERR) & (me <= med + MAD_K * mad)
    return df[keep].sort_values("mjd").reset_index(drop=True)


def main_body(df):
    """Split on adaptive time gaps; keep the brightest eligible segment.
       This drops far-isolated chunks (pre-SN junk, late strays) of any age."""
    if len(df) < MIN_SEG_PTS:
        return df

    d = df.sort_values("mjd").reset_index(drop=True)
    cad = np.diff(d["mjd"].to_numpy())
    if len(cad) == 0:
        return d
    thresh = max(GAP_FACTOR * np.median(cad), GAP_MIN_DAYS)
    seg = np.concatenate([[0], np.cumsum(cad > thresh)])
    d = d.assign(_seg=seg)

    flux, _ = mag_to_flux(d["mag"].to_numpy(), d["magerr"].to_numpy())
    d = d.assign(_flux=flux)

    # score each segment by its 2nd-brightest flux (one spike can't win)
    best, best_score = None, -np.inf
    for s, g in d.groupby("_seg"):
        if len(g) < MIN_SEG_PTS:
            continue
        f = np.sort(g["_flux"].to_numpy())
        score = f[-2] if len(f) >= 2 else f[-1]
        if score > best_score:
            best, best_score = s, score

    if best is None:                                   # nothing eligible -> most points
        best = d["_seg"].value_counts().idxmax()
    return d[d["_seg"] == best].drop(columns=["_seg", "_flux"]).reset_index(drop=True)


def fit_stiff_gp(t, flux, fluxerr):
    amp0 = np.var(flux) + 1e-12
    kernel = (C(amp0, (1e-3 * amp0, 1e3 * amp0))
              * Matern(15.0, LS_BOUNDS, nu=1.5)
              + WhiteKernel(np.median(fluxerr) ** 2,
                            (1e-3 * np.median(fluxerr) ** 2, 1e3 * np.median(fluxerr) ** 2)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=fluxerr ** 2,
                                  normalize_y=False, n_restarts_optimizer=0,
                                  random_state=0)
    gp.fit(t.reshape(-1, 1), flux)
    return gp


def sigma_clip_band(t, flux, fluxerr):
    """Iteratively reject points using LEAVE-ONE-OUT residuals: each point is
       compared to a stiff GP fit to all the OTHER points, so a lone bright
       spike is judged against the real trend (and removed) instead of being
       chased by a fit that includes it. Returns keep-mask + final GP."""
    keep = np.ones(len(t), dtype=bool)
    for _ in range(CLIP_MAX_ITER):
        idx = np.where(keep)[0]
        if len(idx) < 4:
            break
        zval = np.full(len(t), 0.0)
        for j in idx:
            mask = keep.copy()
            mask[j] = False
            if mask.sum() < 3:
                continue
            gp = fit_stiff_gp(t[mask], flux[mask], fluxerr[mask])
            mu, sd = gp.predict(t[j:j + 1].reshape(-1, 1), return_std=True)
            zval[j] = (flux[j] - mu[0]) / np.sqrt(fluxerr[j] ** 2 + sd[0] ** 2)
        bad = keep & (np.abs(zval) > CLIP_SIGMA)
        if not bad.any():                               # converged
            break
        keep = keep & ~bad
    gp = fit_stiff_gp(t[keep], flux[keep], fluxerr[keep]) if keep.sum() >= 3 else None
    return keep, gp


def model_peak(gp, t):
    """Peak (time, flux) from the GP mean, searched only within the data span."""
    grid = np.linspace(t.min(), t.max(), 400)
    mu = gp.predict(grid.reshape(-1, 1))
    i = int(np.argmax(mu))
    return grid[i], mu[i], grid, mu


def active_window(grid, mu, ipeak, thresh):
    """Contiguous time interval around the peak where the model stays > thresh."""
    above = mu > thresh
    lo = ipeak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = ipeak
    while hi < len(grid) - 1 and above[hi + 1]:
        hi += 1
    return grid[lo], grid[hi]


# ----------------------------------------------------------------------
# clean one object
# ----------------------------------------------------------------------
def clean_object(target):
    raw = pd.read_csv(os.path.join(CSV_DIR, f"{target}.csv"))
    n_raw = len(raw)

    df = pre_cuts(raw)
    df = main_body(df)
    if df.empty:
        return raw.iloc[0:0], dict(ZTFID=target, n_raw=n_raw, n_clean=0,
                                   verdict="empty", reason="no points after cuts")

    # ---- per-band stiff GP + sigma clip ----
    band_fit = {}        # name -> (sub_df, keep_mask, gp, t, flux, fluxerr)
    for name, filts in (("g", G_FILTERS), ("r", R_FILTERS)):
        sub = df[df["filter"].isin(filts)]
        if len(sub) < 3:
            continue
        t = sub["mjd"].to_numpy()
        flux, fluxerr = mag_to_flux(sub["mag"].to_numpy(), sub["magerr"].to_numpy())
        keep, gp = sigma_clip_band(t, flux, fluxerr)
        if gp is not None:
            band_fit[name] = dict(idx=sub.index.to_numpy(), keep=keep, gp=gp,
                                  t=t, flux=flux, fluxerr=fluxerr)

    if not band_fit:
        return df.iloc[0:0], dict(ZTFID=target, n_raw=n_raw, n_clean=0,
                                  verdict="insufficient", reason="<3 pts in every band")

    # ---- peak from the model; reference = brighter band ----
    peaks = {}
    for name, b in band_fit.items():
        tk = b["t"][b["keep"]]
        tpk, fpk, grid, mu = model_peak(b["gp"], tk)
        peaks[name] = dict(tpk=tpk, fpk=fpk, grid=grid, mu=mu,
                           mederr=np.median(b["fluxerr"][b["keep"]]))
    ref = max(peaks, key=lambda n: peaks[n]["fpk"])
    t_peak = peaks[ref]["tpk"]

    # cross-band corroboration: do the two bands peak near each other?
    corroborated = True
    if len(peaks) == 2:
        other = "r" if ref == "g" else "g"
        corroborated = abs(peaks[ref]["tpk"] - peaks[other]["tpk"]) <= 30.0

    # ---- active window from the reference-band model ----
    p = peaks[ref]
    thresh = max(WIN_FLUX_FRAC * p["fpk"], WIN_NOISE_K * p["mederr"])
    ipeak = int(np.argmax(p["mu"]))
    win_lo, win_hi = active_window(p["grid"], p["mu"], ipeak, thresh)

    # ---- assemble the global keep set ----
    drop_idx = set()
    for name, b in band_fit.items():
        drop_idx.update(b["idx"][~b["keep"]])          # sigma-clipped points
    keep_mask = ~df.index.isin(drop_idx)
    keep_mask &= (df["mjd"] >= win_lo) & (df["mjd"] <= win_hi)   # active window
    clean = df[keep_mask].reset_index(drop=True)

    # ---- coverage gate ----
    n_g = (clean["filter"].isin(G_FILTERS)).sum()
    n_r = (clean["filter"].isin(R_FILTERS)).sum()
    n_ref = n_g if ref == "g" else n_r
    bracketed = ((clean["mjd"] < t_peak).any() and (clean["mjd"] > t_peak).any())

    if n_ref < MIN_EPOCHS or not bracketed:
        verdict, reason = "insufficient", f"ref {ref}: {n_ref} epochs, bracketed={bracketed}"
    elif not corroborated:
        verdict, reason = "check_peak", "g/r peaks disagree (>30 d)"
    elif n_g >= MIN_COLOR and n_r >= MIN_COLOR:
        verdict, reason = "ok", "good for colour"
    else:
        verdict, reason = "ok_single", "ok but one band sparse for colour"

    report = dict(ZTFID=target, n_raw=n_raw, n_clean=len(clean),
                  n_removed=n_raw - len(clean), ref_band=ref,
                  t_peak_mjd=round(float(t_peak), 2),
                  win_lo=round(float(win_lo), 1), win_hi=round(float(win_hi), 1),
                  n_g=int(n_g), n_r=int(n_r),
                  corroborated=bool(corroborated), verdict=verdict, reason=reason)
    return clean, report


# ----------------------------------------------------------------------
# batch
# ----------------------------------------------------------------------
def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    print(f"Cleaning {len(files)} objects -> {CLEAN_DIR}")

    rows = []
    for f in files:
        target = os.path.splitext(os.path.basename(f))[0]
        try:
            clean, rep = clean_object(target)
        except Exception as e:
            rep = dict(ZTFID=target, verdict="error", reason=str(e)[:80])
            clean = pd.read_csv(f).iloc[0:0]
        clean.to_csv(os.path.join(CLEAN_DIR, f"{target}.csv"), index=False)
        rows.append(rep)

    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(REPORT, index=False)
    print(f"Report -> {REPORT}")
    print(rep_df["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()