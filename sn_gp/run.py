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
import metrics as metrics_mod
import plotting
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF info/warning/error spam
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # don't even probe for a GPU
warnings.filterwarnings("ignore")     # quiet optimizer chatter for now


def fit_object(name, kernel_name, mean_name, gri=False):
    obj = config.load_object(name)

    # ---- standardize the two input axes (store transforms to map back) ----
    tstd = config.Standardizer(obj["t"])
    wstd = config.Standardizer(obj["w"])
    Xt = tstd.forward(obj["t"])
    Xw = wstd.forward(obj["w"])
    X = np.column_stack([Xt, Xw]).astype(np.float64)
    Y = obj["y"].reshape(-1, 1).astype(np.float64)

    # standardized wavelength of each band (blue->red), for the mean functions
    band_waves_std = wstd.forward([config.WAVE_EFF_UM[b] for b in obj["bands"]])

    # ---- lambda policy: free if enough bands, else fixed ----
    free_lambda = obj["n_bands"] >= config.N_BANDS_FREE_LAMBDA
    kernel = kernels.build_kernel(kernel_name, free_lambda)
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

    plotting.plot_fit(obj, predict_slice, m, kernel_name, mean_name, gri=gri)
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="ZTF id, e.g. ZTF25acemaph")
    ap.add_argument("--kernel", required=True,
                    choices=["se", "matern32", "matern52", "rq", "rq_se", "changepoint"])
    ap.add_argument("--mean_func", required=True,
                    choices=["constant", "polynomial", "bazin"])
    ap.add_argument("--gri", action="store_true",
                    help="show only the six g/r/i bands on a 3x2 grid")
    
    args = ap.parse_args()
    fit_object(args.name, args.kernel, args.mean_func, gri=args.gri)


if __name__ == "__main__":
    main()
