"""
metrics.py  --  model-selection numbers for a fitted gpflow model.

k   = number of freely-optimized parameters (kernel + mean + noise), counted
      identically across models for a fair comparison.
lnL = log marginal likelihood at the optimum (model.log_marginal_likelihood()).
n   = total number of data points across ALL bands (one joint model).

AIC = 2k - 2 lnL
BIC = k ln(n) - 2 lnL     (penalizes complexity harder for large n)
"""

import numpy as np
import tensorflow as tf
import gpflow


def count_trainable_params(model):
    """Total scalar count of all trainable parameters in the model."""
    k = 0
    for p in model.trainable_parameters:
        k += int(np.prod(p.shape)) if len(p.shape) else 1
    return k


def compute_aic_bic(model, n):
    lnL = float(model.log_marginal_likelihood().numpy())
    k = count_trainable_params(model)
    aic = 2 * k - 2 * lnL
    bic = k * np.log(n) - 2 * lnL
    return dict(k=k, lnL=lnL, n=int(n), AIC=aic, BIC=bic)