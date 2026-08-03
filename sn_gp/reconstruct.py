"""
reconstruct.py  --  rebuild a fitted GP from its fit/json record and predict.

The measurement pipeline (FWHM, rise/fade, peaks, colour) all read the GP
posterior MEAN, so we must rebuild the *exact* fitted model. This mirrors
run.py's fit_object setup step-by-step, then -- instead of optimising --
assigns the saved parameter values. The standardizers are rebuilt from the
SAVED t_mean/t_sd/w_mean/w_sd (not recomputed), so phase<->standardized maps
back exactly.

Usage:
    from reconstruct import load_fit
    f = load_fit("~/GP_SN/sn_gp/fit/json/ZTF19abfvnns_gibbs_constant.json")
    tg, mu, sd = f.predict_band("ztfr", n=1000)     # flux (mJy) on a fine phase grid
    # tg is phase [days], mu/sd are flux mean/std

Validate before trusting it:
    python reconstruct.py --name ZTF19abfvnns --kernel gibbs --mean constant
  -> writes validate_<...>.png ; compare it to fit/fig/<...>.png by eye.
"""

import os
import json
import argparse
import numpy as np
import gpflow

import config
import kernels
import means


class _Std:
    """Standardizer rebuilt from saved numbers; matches config.Standardizer API
       (has .mu, .sd, forward, inverse)."""
    def __init__(self, mu, sd):
        self.mu = float(mu)
        self.sd = float(sd) or 1.0
    def forward(self, x):  return (np.asarray(x, dtype=float) - self.mu) / self.sd
    def inverse(self, x):  return np.asarray(x, dtype=float) * self.sd + self.mu


class Fit:
    def __init__(self, rec, model, tstd, wstd, obj):
        self.rec, self.model = rec, model
        self.tstd, self.wstd, self.obj = tstd, wstd, obj

    def predict_grid(self, phase_days, wave_um):
        tg = np.asarray(phase_days, dtype=float)
        Xs = np.column_stack([self.tstd.forward(tg),
                              np.full_like(tg, self.wstd.forward(wave_um))])
        mu, var = self.model.predict_f(Xs)
        return tg, np.array(mu).ravel(), np.sqrt(np.array(var).ravel())

    def predict_band(self, band, n=1000, pad=0.0):
        """Predict on a fine phase grid spanning the (post-cut) data of THIS fit.
           band must be a filter name with a known effective wavelength."""
        wave = config.WAVE_EFF_UM[band]
        t = self.obj["t"]
        tg = np.linspace(t.min() - pad, t.max() + pad, n)
        return self.predict_grid(tg, wave)

    def band_data(self, band):
        """Raw (post-cut) data points for a band: (phase, flux, fluxerr)."""
        df = self.obj["df"]
        sub = df[df["filter"] == band]
        return (sub["phase"].to_numpy(), sub["flux"].to_numpy(),
                sub["fluxerr"].to_numpy())


def _rebuild_obj(rec):
    """Reload the object and re-apply the SAME cuts run.py used, so X/Y match."""
    obj = config.load_object(rec["name"])
    t = obj["t"]
    mask = np.ones(len(t), dtype=bool)
    if rec.get("left")  is not None: mask &= (t >= -float(rec["left"]))
    if rec.get("right") is not None: mask &= (t <=  float(rec["right"]))
    # interior gap (only if it was actually used; None -> skip, matches run.py)
    if rec.get("gap_l") is not None and rec.get("gap_r") is not None:
        gl, gr = float(rec["gap_l"]), float(rec["gap_r"])
        mask &= ~((t > gl) & (t < gr))
    for key in ("t", "w", "y", "yerr"):
        obj[key] = obj[key][mask]
    obj["df"] = obj["df"].iloc[mask].reset_index(drop=True)
    obj["bands"] = sorted(obj["df"]["filter"].unique(),
                          key=lambda f: config.WAVE_EFF_UM[f])
    obj["n_bands"] = len(obj["bands"])
    return obj


def load_fit(path):
    rec = json.load(open(os.path.expanduser(path)))
    obj = _rebuild_obj(rec)

    # standardizers from SAVED numbers (exact map back to days/microns)
    tstd = _Std(rec["t_mean"], rec["t_sd"])
    wstd = _Std(rec["w_mean"], rec["w_sd"])

    X = np.column_stack([tstd.forward(obj["t"]), wstd.forward(obj["w"])]).astype(np.float64)
    Y = obj["y"].reshape(-1, 1).astype(np.float64)

    band_waves_std = wstd.forward([config.WAVE_EFF_UM[b] for b in obj["bands"]])
    free_lambda = bool(rec["free_lambda"])

    # build the SAME kernel structure run.py built (so parameter paths match)
    kernel = kernels.build_kernel(rec["kernel"], free_lambda, cp_locs=None)
    # (we don't need cp bounding / t_peak here -- multiple_assign restores the
    #  actual fitted values below; the structure/paths are what must match.)
    mean = means.build_mean(rec["mean"], obj["n_bands"], band_waves_std)

    model = gpflow.models.GPR(
        data=(X, Y), kernel=kernel, mean_function=mean,
        noise_variance=float(np.median(obj["yerr"] ** 2)),
    )

    # restore every fitted parameter by path (constrained values)
    saved = {k: np.array(v) for k, v in rec["params"].items()}
    gpflow.utilities.multiple_assign(model, saved)

    return Fit(rec, model, tstd, wstd, obj)


# ---------------------------------------------------------------- validation
def _validate(name, kernel, mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(config.JSON_DIR, f"{name}_{kernel}_{mean}.json")
    f = load_fit(path)
    obj = f.obj
    print(f"loaded {name} {kernel} {mean}: bands={obj['bands']}, "
          f"n_pts={len(obj['t'])}")

    colors = {"g": "green", "r": "red", "i": "saddlebrown"}
    def _letter(b):
        n = b.split("::")[-1]
        for p in ("sdss", "ztf", "atlas"):
            if n.startswith(p): n = n[len(p):]
        return n

    fig, ax = plt.subplots(figsize=(7, 4))
    for b in obj["bands"]:
        col = colors.get(_letter(b), "grey")
        tg, mu, sd = f.predict_band(b, n=600)
        ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.2)
        ax.plot(tg, mu, color=col, lw=1.2, label=b)
        ph, fl, fe = f.band_data(b)
        ax.errorbar(ph, fl, yerr=fe, fmt="o", ms=3, color=col,
                    mfc="none", elinewidth=0.6, capsize=1.5)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("phase [days]"); ax.set_ylabel("flux [mJy]")
    ax.set_title(f"{name}  {kernel}+{mean}")
    ax.legend(fontsize=7)
    out = f"validate_{name}_{kernel}_{mean}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    print(f"compare to: {os.path.join(config.FIG_DIR, f'{name}_{kernel}_{mean}.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--mean", default="constant")
    a = ap.parse_args()
    _validate(a.name, a.kernel, a.mean)