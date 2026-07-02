"""reconstruct.py -- rebuild a fitted GP from its json record and predict.
   Usage:
     from reconstruct import load_fit
     f = load_fit("~/GP_SN/sn_gp/fit/json/ZTF..._gibbs_constant.json")
     tg, mu, sd = f.predict_band("ztfr", n=300)   # flux grid you can plot however
"""
import os, json, numpy as np, gpflow
import config, kernels, means

class Fit:
    def __init__(self, rec, model, tstd, wstd):
        self.rec, self.model, self.tstd, self.wstd = rec, model, tstd, wstd
    def predict_grid(self, phase_days, wave_um):
        Xt = self.tstd.forward(np.asarray(phase_days, float))
        Xw = self.wstd.forward(np.full_like(Xt, wave_um, dtype=float))
        Xs = np.column_stack([Xt, Xw])
        mu, var = self.model.predict_f(Xs)
        return np.array(mu).ravel(), np.sqrt(np.array(var).ravel())
    def predict_band(self, band, n=300, pad=5.0):
        rec = self.rec
        wave = config.WAVE_EFF_UM[band]
        lo, hi = rec.get("_tmin", -20), rec.get("_tmax", 120)
        tg = np.linspace(lo, hi, n)
        mu, sd = self.predict_grid(tg, wave)
        return tg, mu, sd

def load_fit(path):
    rec = json.load(open(os.path.expanduser(path)))
    obj = config.load_object(rec["name"], gri=rec["gri"])
    # re-apply the SAME mask so training data matches
    # (left/right/gap logic identical to run.py -- factor it out if you like)
    # build standardizers from SAVED params (not recomputed!) for exactness:
    class S:  # minimal standardizer matching config.Standardizer interface
        def __init__(s, m, sd): s.mean, s.sd = m, sd
        def forward(s, x): return (np.asarray(x, float) - s.mean) / s.sd
        def inverse(s, y): return np.asarray(y, float) * s.sd + s.mean
    tstd = S(rec["t_mean"], rec["t_sd"]); wstd = S(rec["w_mean"], rec["w_sd"])
    X = np.column_stack([tstd.forward(obj["t"]), wstd.forward(obj["w"])])
    Y = obj["flux"][:, None]
    kernel = kernels.build_kernel(rec["kernel"], rec["free_lambda"])
    mean = means.build_mean(rec["mean"], obj)   # however you build means
    model = gpflow.models.GPR((X, Y), kernel=kernel, mean_function=mean)
    gpflow.utilities.multiple_assign(model, {k: np.array(v) for k, v in rec["params"].items()})
    return Fit(rec, model, tstd, wstd)
