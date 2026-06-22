"""
config.py  --  all configuration in one place.

Holds: paths, the filter -> effective-wavelength table (PASTE YOUR VALUES BELOW),
band groupings, the lambda-length-scale policy, axis standardization, the AB
flux conversion, and the per-object data loader.

Nothing here fits a GP; it only prepares inputs.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
BASE    = "/users/ariywagh/GP_SN/"
CSV_DIR = os.path.join(BASE, "BTS_csv")

# ----------------------------------------------------------------------
# effective wavelengths  [Angstrom]   <-- PASTE YOUR sncosmo VALUES HERE
# (ztf_x.wave_eff returns Angstroms; keep that unit, we convert to microns below)
# ----------------------------------------------------------------------
WAVE_EFF_ANG = {
    "atlaso":      6866.263838627909,
    "sdssg":       4717.598249924572,
    "sdssr":       6186.798254522246,
    "sdssi":       7506.207753315575,
    "sdssu":       3594.3253668988164,
    "sdssz":       8918.301484406344,
    "uvot::b":     4359.045382941037,
    "uvot::u":     3475.4790196715494,
    "uvot::uvm2":  2254.857934282662,
    "uvot::uvw1":  2614.0745075356726,
    "uvot::uvw2":  2079.0204584448134,
    "uvot::v":     5430.120690049585,
    "ztfg":        4813.948048935322,
    "ztfr":        6421.814890148709,
    "ztfi":        7883.027212349599,
}
# convenience: wavelengths in microns (what the GP actually uses)
WAVE_EFF_UM = {f: (w / 1e4) for f, w in WAVE_EFF_ANG.items() if w is not None}

# bands that count as "g-like" / "r-like" for the brightest-peak reference
G_FILTERS = ["ztfg", "sdssg"]
R_FILTERS = ["ztfr", "sdssr"]

# the six bands shown when --gri is passed (g, r, i for ztf + sdss)
GRI_FILTERS = ["sdssg", "ztfg", "sdssr", "ztfr", "sdssi", "ztfi"]
# ----------------------------------------------------------------------
# lambda length-scale policy (see notes in run.py / chat)
# ----------------------------------------------------------------------
# If an object has >= this many distinct bands, let ell_lambda be a FREE
# hyperparameter; otherwise FIX it to LAMBDA_SCALE_FIXED.
N_BANDS_FREE_LAMBDA = 4
# Fixed value [microns] for sparse objects: ~ the g-r separation (0.147 um).
# Large enough to couple neighbouring bands, small enough to allow real colour.
LAMBDA_SCALE_FIXED = 0.15

# ----------------------------------------------------------------------
# AB flux conversion (work in flux, with a 3% systematic floor)
# ----------------------------------------------------------------------
F0_mJy    = 3631e3       # AB zero-point flux density [mJy]
ERR_FLOOR = 0.03         # systematic fractional flux-error floor

def mag_to_flux(mag, magerr):
    flux = F0_mJy * 10.0 ** (-mag / 2.5)
    fluxerr = flux * (np.log(10.0) / 2.5) * magerr
    fluxerr = np.sqrt(fluxerr ** 2 + (ERR_FLOOR * flux) ** 2)
    return flux, fluxerr


# ----------------------------------------------------------------------
# axis standardization (z-score). Keeps t [days] and lambda [um] on the same
# numeric footing so one kernel length scale isn't straddling very different
# ranges. We store the transforms so predictions can be mapped back.
# ----------------------------------------------------------------------
class Standardizer:
    def __init__(self, values):
        self.mu = float(np.mean(values))
        self.sd = float(np.std(values)) or 1.0
    def forward(self, x):   return (np.asarray(x) - self.mu) / self.sd
    def inverse(self, x):   return np.asarray(x) * self.sd + self.mu


# ----------------------------------------------------------------------
# data loader: one object -> arrays ready for the MOGP
# ----------------------------------------------------------------------
def load_object(name):
    """Returns a dict with raw arrays and metadata. Drops alert_fp only
       (no manual cuts, no isolation algorithm). Flux in mJy; phase = MJD - peak,
       peak = brightest point across the g/r bands (the reference)."""
    df = pd.read_csv(os.path.join(CSV_DIR, f"{name}.csv"))
    df = df[df["origin"] != "alert_fp"].copy()

    # keep only filters we have a wavelength for
    df = df[df["filter"].isin(WAVE_EFF_UM.keys())].copy()
    if df.empty:
        raise ValueError(f"{name}: no points in filters with known wavelengths")

    flux, fluxerr = mag_to_flux(df["mag"].to_numpy(), df["magerr"].to_numpy())
    df["flux"], df["fluxerr"] = flux, fluxerr
    df["wave_um"] = df["filter"].map(WAVE_EFF_UM)

    # reference peak = brightest (max flux) among g/r bands
    gr = df[df["filter"].isin(G_FILTERS + R_FILTERS)]
    ref_df = gr if not gr.empty else df
    peak_mjd = ref_df.loc[ref_df["flux"].idxmax(), "mjd"]
    df["phase"] = df["mjd"] - peak_mjd

    df = df.sort_values("phase").reset_index(drop=True)
    bands = sorted(df["filter"].unique(), key=lambda f: WAVE_EFF_UM[f])  # blue -> red

    return dict(
        name=name, df=df, bands=bands,
        t=df["phase"].to_numpy(),
        w=df["wave_um"].to_numpy(),
        y=df["flux"].to_numpy(),
        yerr=df["fluxerr"].to_numpy(),
        peak_mjd=float(peak_mjd),
        n_bands=len(bands),
    )