"""
run.py  --  fit ONE object with ONE (kernel, mean) pair and plot it.

    python3 run.py --name ZTF25acemaph --kernel matern32 --mean_func constant --gri

Group 1 (vary kernel, mean=constant): loop kernels yourself:
    for K in se matern32 matern52 rq rq_se changepoint; do
        python3 run.py --name <obj> --kernel $K --mean_func constant; done
Group 2 (vary mean, kernel=matern52):
    for M in constant polynomial bazin; do
        python3 run.py --name <obj> --kernel matern52 --mean_func $M; done
"""

import argparse
import warnings
import numpy as np
import tensorflow as tf
import gpflow

import config
import kernels
import means
import json
import metrics as metrics_mod
import plotting
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF info/warning/error spam
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # don't even probe for a GPU
warnings.filterwarnings("ignore")     # quiet optimizer chatter for now


def fit_object(name, kernel_name, mean_name, gri=False, left=None, right=None,
               gap_l=None, gap_r=None, cp_days=None):
    obj = config.load_object(name)
    t = obj["t"]
    mask = np.ones(len(t), dtype=bool)
    if left  is not None: mask &= (t >= -left)
    if right is not None: mask &= (t <=  right)
    # subset every per-point array consistently
    for key in ("t", "w", "y", "yerr"):
        obj[key] = obj[key][mask]
    obj["df"] = obj["df"].iloc[mask].reset_index(drop=True)
    obj["bands"] = sorted(obj["df"]["filter"].unique(),
                          key=lambda f: config.WAVE_EFF_UM[f])
    obj["n_bands"] = len(obj["bands"])

    # ---- standardize the two input axes (store transforms to map back) ----

    tstd = config.Standardizer(obj["t"])
    wstd = config.Standardizer(obj["w"])
    Xt = tstd.forward(obj["t"])
    Xw = wstd.forward(obj["w"])
    X = np.column_stack([Xt, Xw]).astype(np.float64)
    Y = obj["y"].reshape(-1, 1).astype(np.float64)


    cp_locs_std = None          # for changepoint kernels
    gibbs_peak_std = 0.0        # default Gibbs anchor = phase 0 (peak)
    if cp_days is not None:
        d1, d2 = cp_days
        if d1 is not None:
            cp_locs_std = [float(tstd.forward(np.array([d1]))[0])]
            gibbs_peak_std = float(tstd.forward(np.array([d1]))[0])   # re-anchor Gibbs to the real peak
        if d2 is not None:
            cp_locs_std.append(float(tstd.forward(np.array([d2]))[0]))

    # standardized wavelength of each band (blue->red), for the mean functions
    band_waves_std = wstd.forward([config.WAVE_EFF_UM[b] for b in obj["bands"]])

    # ---- lambda policy: free if enough bands, else fixed ----
    free_lambda = obj["n_bands"] >= config.N_BANDS_FREE_LAMBDA
    # kernel = kernels.build_kernel(kernel_name, free_lambda)

    kernel = kernels.build_kernel(kernel_name, free_lambda, cp_locs=cp_locs_std)

    # bound changepoint locations so they can't wander off-data
    if kernel_name in ("changepoint", "changepoint_1"):
        cp = kernel.kernels[0].base
        kernels.bound_cp_locations(
            cp,
            lo=float(tstd.forward(np.array([-10.0]))[0]),
            hi=float(tstd.forward(np.array([ 10.0]))[0]),
        )
    # re-anchor Gibbs dip to the (possibly overridden) peak
    if kernel_name == "gibbs":
        gk = kernel.kernels[0].base                 # _TimeOnly wraps GibbsTime
        gk.t_peak.assign(gibbs_peak_std)
    
    # set the wavelength length scale (standardized units); fix it if sparse
    k_wave = kernel.kernels[1]
    k_wave.lengthscales.assign(config.LAMBDA_SCALE_FIXED / wstd.sd)
    if not free_lambda:
        gpflow.set_trainable(k_wave.lengthscales, False)

    mean = means.build_mean(mean_name, obj["n_bands"], band_waves_std)

    # ---- model: per-point measurement errors enter as a fixed noise diagonal;
    #      gpflow's likelihood variance acts as extra (tiny) jitter ----
    model = gpflow.models.GPR(
        data=(X, Y), kernel=kernel, mean_function=mean,
        noise_variance=float(np.median(obj["yerr"] ** 2)),
    )
    # bake per-point variances in by scaling: GPR uses a scalar noise, so we
    # approximate heteroscedastic errors by adding them to the kernel diagonal
    # via a custom likelihood is overkill here; instead we keep the scalar noise
    # and rely on the 3% floor. (For strict per-point errors, switch to a
    # heteroscedastic likelihood later.)

    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables,
        options=dict(maxiter=500),
    )

    m = metrics_mod.compute_aic_bic(model, n=len(obj["y"]))

# write a machine-readable sidecar so the batch driver can collect metrics
    def _params(mdl):
        return {path: np.array(p).tolist()
                for path, p in gpflow.utilities.parameter_dict(mdl).items()}

    os.makedirs(config.JSON_DIR, exist_ok=True)
    record = dict(
        name=name, kernel=kernel_name, mean=mean_name, gri=bool(gri),
        left=left, right=right, gap_l=gap_l, gap_r=gap_r,
        cp_days=list(cp_days) if cp_days else None,
        t_mean=float(tstd.mu), t_sd=float(tstd.sd),
        w_mean=float(wstd.mu), w_sd=float(wstd.sd),
        free_lambda=bool(free_lambda),
        bands=list(obj["bands"]),
        k=m["k"], lnL=m["lnL"], n=m["n"], AIC=m["AIC"], BIC=m["BIC"],
        params=_params(model),
    )
    with open(os.path.join(config.JSON_DIR,
              f"{name}_{kernel_name}_{mean_name}.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n{name}  kernel={kernel_name}  mean={mean_name}  "
          f"(free_lambda={free_lambda})")
    print(f"  k={m['k']}  lnL={m['lnL']:.2f}  AIC={m['AIC']:.2f}  BIC={m['BIC']:.2f}")
    gpflow.utilities.print_summary(model)

    # ---- prediction slice for one band (returns DATA units) ----
    def predict_slice(band, wave_um):
        tg = np.linspace(obj["t"].min() - 5, obj["t"].max() + 5, 300)
        Xs = np.column_stack([tstd.forward(tg),
                              np.full_like(tg, wstd.forward(wave_um))])
        mu, var = model.predict_f(Xs)
        return tg, mu.numpy().ravel(), np.sqrt(var.numpy().ravel())

    t0_phase = None
    if hasattr(model.mean_function, "t0"):
        t0_std = float(np.ravel(model.mean_function.t0.numpy())[0])
        t0_phase = tstd.inverse(t0_std)        # standardized time -> phase [days]

    plotting.plot_fit(obj, predict_slice, m, kernel_name, mean_name,
                      gri=gri, t0_phase=t0_phase, outdir=config.FIG_DIR)
    return model, m



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="ZTF id, e.g. ZTF25acemaph")
    ap.add_argument("--kernel", required=True,
        choices=["se", "matern32", "matern52", "rq", "rq_se", "changepoint", "gibbs", "changepoint_1"])
    ap.add_argument("--mean_func", required=True,
                    choices=["constant", "polynomial", "bazin"])
    ap.add_argument("--gri", action="store_true",
                    help="show only the six g/r/i bands on a 3x2 grid")
    ap.add_argument("--left",  type=float, default=None,
                help="keep phase >= peak - left (accepts negatives)")
    ap.add_argument("--right", type=float, default=None,
                    help="keep phase <= peak + right (accepts negatives)")  
    ap.add_argument("--outdir", default="figs",
                    help="subfolder under sn_gp/ for the PNG + JSON (default figs)") 
    ap.add_argument("--gap_l", type=float, default=None,
                    help="left edge (phase, days) of an interior region to EXCLUDE from the fit")
    ap.add_argument("--gap_r", type=float, default=None,
                    help="right edge (phase, days) of the excluded region")  
    ap.add_argument("--cp_loc", default=None,
        help="changepoint location(s) in PHASE DAYS, comma-separated "
             "(e.g. '3' or '-1,6'). Overrides kernel default; also re-anchors Gibbs.")
    
    args = ap.parse_args()
    cp_days = None
    if args.cp_loc:
        parts = [float(x) for x in args.cp_loc.split(",")]
        cp_days = (parts[0], parts[1] if len(parts) > 1 else None)
    fit_object(args.name, args.kernel, args.mean_func, gri=args.gri,
               left=args.left, right=args.right,
               gap_l=args.gap_l, gap_r=args.gap_r, cp_days=cp_days)


if __name__ == "__main__":
    main()
