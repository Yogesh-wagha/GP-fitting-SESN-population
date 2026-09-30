"""
fit_mag_villar.py  --  standalone: fit ONE light curve in MAGNITUDE space with a
Matern-3/2 kernel and a per-band Villar mean, then measure and plot it.

Independent of the flux pipeline (run.py / measure_population.py); writes its
own PNG + JSON and prints every measured parameter.

    python fit_mag_villar.py --name ZTF21abieuta
    python fit_mag_villar.py --name ZTF21abieuta --left 20 --right 120
    python fit_mag_villar.py --name ZTF21abieuta --mean constant   # compare means

NOTE on magnitude space: mag is undefined for flux <= 0, so non-detections and
negative-flux points are dropped. Peak = MINIMUM magnitude; the half-flux width
is the width at peak + 0.7526 mag.
"""

import os, json, argparse, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from gpflow import Parameter
from gpflow.functions import MeanFunction
from astropy.cosmology import FlatLambdaCDM

import config

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
BTS = os.path.expanduser("~/GP_SN/BTS.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "mag_villar")
os.makedirs(OUTDIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

HALF_FLUX_MAG = 2.5 * np.log10(2.0)      # 0.7526 mag = half flux
MIN_BAND_POINTS = 5
GRI = ("g", "r", "i")
BAND_COLOR = {"g": "#1b9e2f", "r": "#d62728", "i": "#8c564b"}


def _letter(b):
    n = str(b).split("::")[-1]
    for p in ("sdss", "ztf", "atlas", "ps1"):
        if n.startswith(p):
            n = n[len(p):]
    return n[:1].lower() if n else ""


# ------------------------------------------------------------------ mean fns
def _band_index(Xwave, band_waves):
    bw = tf.constant(band_waves, dtype=Xwave.dtype)
    return tf.argmin(tf.abs(Xwave[:, None] - bw[None, :]), axis=1)


class ConstantPerBand(MeanFunction):
    def __init__(self, n_bands, band_waves, init=None):
        super().__init__()
        self.band_waves = band_waves
        self.c = Parameter(np.zeros(n_bands) if init is None else np.asarray(init))
    def __call__(self, X):
        return tf.gather(self.c, _band_index(X[:, 1], self.band_waves))[:, None]


class VillarPerBand(MeanFunction):
    """Villar-style, in MAGNITUDE space:

        m_b(t) = [ y0 + beta*(t-t0) + g0*exp(-(t-t0)^2 / (2 sigma^2)) ]
                 / [ 1 + exp(-(t - tau)/theta) ]

    In magnitudes the Gaussian term is NEGATIVE at peak (brighter = smaller
    mag), so g0 is constrained <= 0; beta >= 0 gives a fading linear decline.
    """
    def __init__(self, n_bands, band_waves, y0_init=None):
        super().__init__()
        self.band_waves = band_waves
        pos = gpflow.utilities.positive()
        self.y0       = Parameter(np.zeros(n_bands) if y0_init is None
                                  else np.asarray(y0_init))
        self.beta_pos = Parameter(np.ones(n_bands) * 0.05, transform=pos)  # fade
        self.g0_pos   = Parameter(np.ones(n_bands) * 1.0,  transform=pos)  # depth
        self.t0       = Parameter(np.zeros(n_bands))
        self.sigma    = Parameter(np.ones(n_bands),        transform=pos)
        self.tau      = Parameter(np.zeros(n_bands))
        self.theta    = Parameter(np.ones(n_bands),        transform=pos)

    def __call__(self, X):
        t = X[:, 0]
        idx = _band_index(X[:, 1], self.band_waves)
        y0 = tf.gather(self.y0, idx)
        b  = tf.gather(self.beta_pos, idx)          # >= 0 : fading
        g0 = -tf.gather(self.g0_pos, idx)           # <= 0 : brightening at peak
        t0 = tf.gather(self.t0, idx)
        s  = tf.gather(self.sigma, idx)
        ta = tf.gather(self.tau, idx)
        th = tf.gather(self.theta, idx)
        dt = t - t0
        num = y0 + b * dt + g0 * tf.exp(-0.5 * (dt / s) ** 2)
        z = tf.clip_by_value(-(t - ta) / th, -30.0, 30.0)
        return (num / (1.0 + tf.exp(z)))[:, None]


# ------------------------------------------------------------------ data
def load_mag(name, left=None, right=None, bands_mode="gri"):
    """Object in MAGNITUDE space; drops non-positive flux (mag undefined)."""
    obj = config.load_object(name)
    df = obj["df"].copy()
    if "mag" not in df.columns:                       # derive if absent
        F0 = 3631e3
        df = df[df["flux"] > 0]
        df["mag"] = -2.5 * np.log10(df["flux"] / F0)
        df["magerr"] = (2.5 / np.log(10)) * df["fluxerr"] / df["flux"]
    df = df[np.isfinite(df["mag"]) & np.isfinite(df["magerr"])]
    t = df["phase"].to_numpy()
    m = np.ones(len(df), dtype=bool)
    if left  is not None: m &= (t >= -float(left))
    if right is not None: m &= (t <=  float(right))
    if bands_mode == "gri":
        m &= np.isin([_letter(f) for f in df["filter"]], list(GRI))
    filt = df["filter"].to_numpy()
    counts = {}
    for f in filt[m]:
        counts[f] = counts.get(f, 0) + 1
    sparse = [f for f, c in counts.items() if c < MIN_BAND_POINTS]
    if sparse:
        print(f"  dropping sparse bands (<{MIN_BAND_POINTS}): "
              + ", ".join(f"{f}({counts[f]})" for f in sorted(sparse)))
        m &= ~np.isin(filt, sparse)
    df = df[m].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"{name}: nothing left after cuts")
    bands = sorted(df["filter"].unique(), key=lambda f: config.WAVE_EFF_UM[f])
    return dict(df=df, bands=bands, n_bands=len(bands),
                t=df["phase"].to_numpy(),
                w=np.array([config.WAVE_EFF_UM[f] for f in df["filter"]]),
                y=df["mag"].to_numpy(), yerr=df["magerr"].to_numpy())


def lookup_meta(ztfid):
    bts = pd.read_csv(BTS)
    idc = "ZTFID" if "ZTFID" in bts.columns else bts.columns[0]
    zc = "redshift" if "redshift" in bts.columns else "z"
    row = bts[bts[idc].astype(str) == ztfid]
    if row.empty:
        return "", np.nan, np.nan
    r = row.iloc[0]
    def num(v):
        try:
            x = float(v); return x if x > 0 else np.nan
        except (TypeError, ValueError):
            return np.nan
    return str(r.get("type", "")), num(r.get(zc)), num(r.get("A_V"))


# ------------------------------------------------------------------ fit
def fit(name, left, right, mean_name, kernel_name="matern32", maxiter=1000):
    obj = load_mag(name, left, right)
    tstd = config.Standardizer(obj["t"])
    wstd = config.Standardizer(obj["w"])
    X = np.column_stack([tstd.forward(obj["t"]), wstd.forward(obj["w"])]).astype(np.float64)
    Y = obj["y"].reshape(-1, 1).astype(np.float64)

    bw = wstd.forward([config.WAVE_EFF_UM[b] for b in obj["bands"]])
    # sensible starting point: each band's median magnitude
    med = [float(np.median(obj["df"].loc[obj["df"]["filter"] == b, "mag"]))
           for b in obj["bands"]]

    import kernels as kern_mod
    kernel = kern_mod.build_kernel(kernel_name, free_lambda=False, cp_locs=None)

    # ---- changepoint: bound both locations to +/-10 d around peak ----
    if kernel_name in ("changepoint", "changepoint_1"):
        cp = kernel.kernels[0].base
        kern_mod.bound_cp_locations(
            cp,
            lo=float(tstd.forward(np.array([-10.0]))[0]),
            hi=float(tstd.forward(np.array([ 10.0]))[0]),
        )
    # ---- gibbs: anchor the lengthscale dip at the peak (phase 0) ----
    if kernel_name == "gibbs":
        gk = kernel.kernels[0].base
        gk.t_peak.assign(float(tstd.forward(np.array([0.0]))[0]))

    k_wave = kernel.kernels[1]
    k_wave.lengthscales.assign(config.LAMBDA_SCALE_FIXED / wstd.sd)
    gpflow.set_trainable(k_wave.lengthscales, False)

    mean = (VillarPerBand(obj["n_bands"], bw, y0_init=med) if mean_name == "villar"
            else ConstantPerBand(obj["n_bands"], bw, init=med))

    model = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=mean,
                              noise_variance=float(np.median(obj["yerr"] ** 2)))
    gpflow.optimizers.Scipy().minimize(model.training_loss,
                                       model.trainable_variables,
                                       options=dict(maxiter=maxiter))
    lnL = float(model.log_marginal_likelihood())
    k = int(sum(int(np.size(p)) for p in model.trainable_parameters))
    n = len(Y)
    aic = 2 * k - 2 * lnL
    bic = k * np.log(n) - 2 * lnL
    print(f"\n{name}  {kernel_name}+{mean_name}  [MAGNITUDE space]")
    print(f"  n={n}  k={k}  lnL={lnL:.2f}  AIC={aic:.2f}  BIC={bic:.2f}")
    gpflow.utilities.print_summary(model)
    return model, obj, tstd, wstd, dict(k=k, lnL=lnL, n=n, AIC=aic, BIC=bic)


def predict(model, tstd, wstd, tg, wave):
    Xs = np.column_stack([tstd.forward(tg),
                          np.full_like(tg, wstd.forward(wave))])
    mu, var = model.predict_f(Xs)
    return np.array(mu).ravel(), np.sqrt(np.array(var).ravel())


# ------------------------------------------------------------------ measure
def measure(model, obj, tstd, wstd, z, ngrid=2000):
    """All shape/colour params, computed directly in magnitude space."""
    out = {}
    zf = (1.0 + z) if np.isfinite(z) and z > 0 else np.nan
    per_band = {}
    for b in obj["bands"]:
        sub = obj["df"][obj["df"]["filter"] == b]
        ph = sub["phase"].to_numpy()
        tg = np.linspace(ph.min(), ph.max(), ngrid)
        mu, sd = predict(model, tstd, wstd, tg, config.WAVE_EFF_UM[b])
        ipk = int(np.argmin(mu))                    # PEAK = MINIMUM magnitude
        mpk, tpk = float(mu[ipk]), float(tg[ipk])
        half = mpk + HALF_FLUX_MAG                  # fainter by 0.753 mag
        # rise side: last crossing FAINTER than half, before peak
        tL = np.nan
        above = np.where(mu[:ipk + 1] > half)[0]
        if above.size and above[-1] + 1 <= ipk:
            i0 = above[-1]
            f0, f1, t0, t1 = mu[i0], mu[i0+1], tg[i0], tg[i0+1]
            tL = t0 + (half - f0)*(t1-t0)/(f1-f0) if f1 != f0 else t0
        # fade side: first crossing fainter than half, after peak
        tR = np.nan
        after = np.where(mu[ipk:] > half)[0]
        if after.size and after[0] > 0:
            j = ipk + after[0]
            f0, f1, t0, t1 = mu[j-1], mu[j], tg[j-1], tg[j]
            tR = t0 + (half - f0)*(t1-t0)/(f1-f0) if f1 != f0 else t1
        rise = tpk - tL if np.isfinite(tL) else np.nan
        fade = tR - tpk if np.isfinite(tR) else np.nan
        fwhm = rise + fade if (np.isfinite(rise) and np.isfinite(fade)) else np.nan
        # dm15 (rest-frame 15 d after peak)
        dm15 = np.nan
        t15 = tpk + 15.0 * zf if np.isfinite(zf) else np.nan
        if np.isfinite(t15) and ph.min() <= t15 <= ph.max():
            m15, _ = predict(model, tstd, wstd, np.array([t15]), config.WAVE_EFF_UM[b])
            dm15 = float(m15[0] - mpk)
        # tail slope over +30..+60 rest-frame days
        tail = np.nan; npts = 0
        if np.isfinite(zf):
            lo, hi = max(tpk + 30*zf, ph.min()), min(tpk + 60*zf, ph.max())
            if hi - lo > 5 * zf:
                tt = np.linspace(lo, hi, 300)
                mm, _ = predict(model, tstd, wstd, tt, config.WAVE_EFF_UM[b])
                tail = float(np.polyfit((tt - tpk)/zf, mm, 1)[0])
                npts = int(np.sum((ph >= lo) & (ph <= hi)))
        per_band[b] = dict(peak_mag=mpk, peak_phase=tpk, peak_magerr=float(sd[ipk]),
                           rise_obs=rise, fade_obs=fade, fwhm_obs=fwhm,
                           rise=rise/zf if np.isfinite(zf) else np.nan,
                           fade=fade/zf if np.isfinite(zf) else np.nan,
                           fwhm=fwhm/zf if np.isfinite(zf) else np.nan,
                           dm15=dm15, tail_slope=tail, tail_npts=npts,
                           span=(float(ph.min()), float(ph.max())), n=len(ph))
    out["bands"] = per_band

    # absolute magnitudes + colours (needs a g and an r band)
    gb = next((b for b in obj["bands"] if _letter(b) == "g"), None)
    rb = next((b for b in obj["bands"] if _letter(b) == "r"), None)
    DM = (5*np.log10(cosmo.luminosity_distance(z).to("pc").value/10.0)
          if np.isfinite(z) and z > 0 else np.nan)
    out["DM"] = DM
    for b, tag in ((gb, "g"), (rb, "r")):
        out[f"M_{tag}"] = (per_band[b]["peak_mag"] - DM
                           if b and np.isfinite(DM) else np.nan)
    if gb and rb and np.isfinite(DM):
        tpk_r = per_band[rb]["peak_phase"]
        def mag_at(b, t):
            ph = obj["df"].loc[obj["df"]["filter"] == b, "phase"].to_numpy()
            if not (ph.min() <= t <= ph.max()):
                return np.nan
            m, _ = predict(model, tstd, wstd, np.array([t]), config.WAVE_EFF_UM[b])
            return float(m[0])
        out["color_at_rpeak"] = mag_at(gb, tpk_r) - mag_at(rb, tpk_r)
        t10 = tpk_r + 10.0 * zf if np.isfinite(zf) else np.nan
        out["color_10d"] = (mag_at(gb, t10) - mag_at(rb, t10)
                            if np.isfinite(t10) else np.nan)
        out["color_ownpeak"] = per_band[gb]["peak_mag"] - per_band[rb]["peak_mag"]
        out["dt_peak_g_minus_r"] = ((per_band[gb]["peak_phase"] - tpk_r) / zf
                                    if np.isfinite(zf) else np.nan)
        fg, fr = per_band[gb]["fwhm"], per_band[rb]["fwhm"]
        out["fwhm_ratio_gr"] = fg/fr if (np.isfinite(fg) and np.isfinite(fr) and fr>0) else np.nan
    return out


def report(name, sn_type, z, a_v, met, meas):
    print("\n" + "=" * 62)
    print(f"MEASURED PARAMETERS   {name}   type={sn_type}   z={z}")
    print("=" * 62)
    print(f"  fit: n={met['n']} k={met['k']} lnL={met['lnL']:.2f} "
          f"AIC={met['AIC']:.2f} BIC={met['BIC']:.2f}")
    for b, d in meas["bands"].items():
        print(f"\n  [{b}]  ({d['n']} pts, phase {d['span'][0]:.1f} to {d['span'][1]:.1f})")
        print(f"    peak mag        {d['peak_mag']:.3f} +/- {d['peak_magerr']:.3f} "
              f"at phase {d['peak_phase']:.2f}")
        for lab, key in (("rise (rest)", "rise"), ("fade (rest)", "fade"),
                         ("FWHM (rest)", "fwhm"), ("dm15", "dm15")):
            v = d[key]
            print(f"    {lab:15s} {v:.3f}" if np.isfinite(v) else f"    {lab:15s} --")
        v = d["tail_slope"]
        print(f"    tail slope      {v:.5f} mag/d  ({d['tail_npts']} pts in window)"
              if np.isfinite(v) else "    tail slope      --")
    print(f"\n  distance modulus  {meas['DM']:.3f}" if np.isfinite(meas.get("DM", np.nan))
          else "\n  distance modulus  --")
    for lab, key in (("M_g", "M_g"), ("M_r", "M_r"),
                     ("g-r at r-peak", "color_at_rpeak"), ("g-r at +10d", "color_10d"),
                     ("g-r own peaks", "color_ownpeak"),
                     ("t_g - t_r [d]", "dt_peak_g_minus_r"),
                     ("FWHM ratio g/r", "fwhm_ratio_gr")):
        v = meas.get(key, np.nan)
        print(f"  {lab:16s} {v:.3f}" if np.isfinite(v) else f"  {lab:16s} --")
    print("=" * 62)


def plot(name, model, obj, tstd, wstd, meas, mean_name, kernel_name):
    fig, ax = plt.subplots(figsize=(9, 6))
    for b in obj["bands"]:
        col = BAND_COLOR.get(_letter(b), "grey")
        sub = obj["df"][obj["df"]["filter"] == b]
        ph = sub["phase"].to_numpy()
        tg = np.linspace(ph.min(), ph.max(), 500)
        mu, sd = predict(model, tstd, wstd, tg, config.WAVE_EFF_UM[b])
        ax.plot(tg, mu, color=col, lw=1.8, label=str(b))
        ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.errorbar(ph, sub["mag"], yerr=sub["magerr"], fmt="o", ms=5,
                    color=col, mfc="white", mec=col, elinewidth=0.9, capsize=2,
                    zorder=3)
        d = meas["bands"][b]
        ax.plot(d["peak_phase"], d["peak_mag"], marker="v", color=col,
                ms=11, mec="k", mew=0.8, zorder=4)
        ax.axhline(d["peak_mag"] + HALF_FLUX_MAG, color=col, ls=":", lw=0.9,
                   alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("phase [days]"); ax.set_ylabel("apparent magnitude (AB)")
    # ax.set_title(f"{name}   {kernel_name} + {mean_name}   [magnitude-space fit]")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    out = os.path.join(OUTDIR, f"{name}_{kernel_name}_{mean_name}_mag.png")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\nwrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mean", default="villar", choices=["villar", "constant"])
    ap.add_argument("--kernel", default="matern32")
    ap.add_argument("--left", type=float, default=None)
    ap.add_argument("--right", type=float, default=None)
    ap.add_argument("--maxiter", type=int, default=1000)
    a = ap.parse_args()

    sn_type, z, a_v = lookup_meta(a.name)
    model, obj, tstd, wstd, met = fit(a.name, a.left, a.right, a.mean,
                                      a.kernel, a.maxiter)
    meas = measure(model, obj, tstd, wstd, z)
    report(a.name, sn_type, z, a_v, met, meas)
    plot(a.name, model, obj, tstd, wstd, meas, a.mean, a.kernel)

    rec = dict(name=a.name, type=sn_type, z=z, A_V=a_v, space="magnitude",
               kernel=a.kernel, mean=a.mean, left=a.left, right=a.right,
               bands=list(obj["bands"]), metrics=met,
               measurements={k: (v if not isinstance(v, dict) else v)
                             for k, v in meas.items()})
    jp = os.path.join(OUTDIR, f"{a.name}_{a.kernel}_{a.mean}_mag.json")
    with open(jp, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()