"""
kernels.py  --  build a 2-D (time x wavelength) gpflow kernel by name.

The model kernel is always   k_time(t) * k_wave(lambda).
- k_time is the named kernel and carries the light-curve-shape length scale.
- k_wave is a Matern-3/2 on the wavelength axis (active_dims=[1]) that couples
  the bands. Its length scale is either free or fixed (see free_lambda).

active_dims tells gpflow which input column a kernel acts on:
  column 0 = time, column 1 = wavelength (both standardized in run.py).

--kernel names: se, matern32, matern52, rq, rq_se, changepoint
"""

import numpy as np
import gpflow
from gpflow.kernels import (SquaredExponential, Matern32, Matern52,
                            RationalQuadratic, Constant, Linear, ChangePoints, Kernel, White)

import tensorflow_probability as tfp
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.utilities import positive

class GibbsTime(Kernel):
    """Non-stationary Gibbs kernel on a 1-D time axis (the lengthscale varies
    with time -- a valid positive-definite generalisation of the SE kernel).

    Lengthscale:
        l(t) = l_min + l_relax * (1 - exp(-((t - t_peak)/width)^2))
      -> l_min at the peak  (a DIP: wiggly, follows fast rise/peak)
      -> l_min + l_relax far from peak  (smooth tail)
    Writing it as l_min + l_relax (both positive) GUARANTEES the dip, since the
    away-from-peak value is always >= the peak value.

    Gibbs covariance:
        k(t,t') = var * sqrt( 2 l(t) l(t') / (l(t)^2 + l(t')^2) )
                      * exp( -(t - t')^2 / (l(t)^2 + l(t')^2) )

    Use it wrapped by _TimeOnly (so it sees only the time column) and multiplied
    by k_wave -- same as the changepoint kernel.

    t_peak is FIXED by default (anchored at phase 0 = peak). Defaults are in
    DAYS, so they are correct if you fit in raw phase. If your time axis is
    standardised, pass t_peak / l_min / l_relax / width in standardised units.
    """
    def __init__(self, variance=1.0, l_min=3.0, l_relax=15.0, width=12.0,
                 t_peak=0.0, train_peak=False):
        super().__init__()
        self.variance = Parameter(variance, transform=positive())
        self.l_min    = Parameter(l_min,    transform=positive())   # lengthscale at peak
        self.l_relax  = Parameter(l_relax,  transform=positive())   # extra length on the tail
        self.width    = Parameter(width,    transform=positive())   # how wide the dip is
        self.t_peak   = Parameter(t_peak)                           # dip location (phase 0)
        gpflow.set_trainable(self.t_peak, train_peak)

    def _lengthscale(self, t):
        z = (t - self.t_peak) / self.width
        return self.l_min + self.l_relax * (1.0 - tf.exp(-tf.square(z)))   # [N]

    def K(self, X, X2=None):
        t  = X[:, 0]                                   # X is [N,1] after _TimeOnly
        t2 = t if X2 is None else X2[:, 0]
        l  = self._lengthscale(t)                      # [N]
        l2 = self._lengthscale(t2)                     # [M]
        l_sq = tf.square(l)[:, None] + tf.square(l2)[None, :] + 1e-12   # [N,M]
        prefac = tf.sqrt(2.0 * l[:, None] * l2[None, :] / l_sq)
        d2 = tf.square(t[:, None] - t2[None, :])
        return self.variance * prefac * tf.exp(-d2 / l_sq)

    def K_diag(self, X):
        # k(t,t): prefactor = 1, exp(0) = 1  ->  variance
        return self.variance * tf.ones(tf.shape(X)[0], dtype=X.dtype)


class _TimeOnly(Kernel):
    """Wraps a kernel so it only ever sees the time column (index 0) of a
       2-D [time, wavelength] input. This lets gpflow's ChangePoints (which
       switches along ONE axis) work inside our 2-D model without the
       reshape error you get from putting active_dims on its sub-kernels."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def K(self, X, X2=None):
        X = X[:, :1]                       # keep time only
        X2 = None if X2 is None else X2[:, :1]
        return self.base.K(X, X2)

    def K_diag(self, X):
        return self.base.K_diag(X[:, :1])


def _time_kernel(name):
    """The time-axis kernel (active_dims=[0])."""
    if name == "se":
        return SquaredExponential(active_dims=[0])
    if name == "matern32":
        return Matern32(active_dims=[0])
    if name == "matern52":
        return Matern52(active_dims=[0])
    if name == "rq":
        return RationalQuadratic(active_dims=[0])
    if name == "rq_se":
        return RationalQuadratic(active_dims=[0]) * SquaredExponential(active_dims=[0])
    if name == "changepoint":
        return _changepoint_time_kernel()
    raise ValueError(f"unknown kernel '{name}'")


def _changepoint_time_kernel(loc1=-0.5, loc2=0.5, steep=1.0):
    """3 segments: Constant-like(Matern32) | Matern32 rise | Linear tail.
       loc1, loc2 in STANDARDIZED time."""
    base = Matern32()
    rise = Matern32()
    tail = Linear()
    return ChangePoints(kernels=[base, rise, tail],
                        locations=[loc1, loc2], steepness=steep)

def _changepoint_time_kernel_1(loc=0.0, steep=1.0):
    """2 segments: Matern32 rise | Matern32 tail. loc in STANDARDIZED time."""
    rise = Matern32()
    tail = Matern32()
    return ChangePoints(kernels=[rise, tail],
                        locations=[loc], steepness=steep)

def bound_cp_locations(cp, lo, hi):
    """Constrain ChangePoints.locations to [lo, hi] (standardized units)."""
    import numpy as np
    lo = np.float64(lo)
    hi = np.float64(hi)
    init = np.clip(cp.locations.numpy(), lo + 1e-3, hi - 1e-3).astype(np.float64)
    cp.locations = Parameter(
        init,
        transform=tfp.bijectors.Sigmoid(low=lo, high=hi),
        dtype=np.float64,
    )

def build_kernel(name, free_lambda, cp_locs=None):
    if name == "gibbs":
        k_time = _TimeOnly(GibbsTime())
    elif name == "changepoint":                       # 3-segment
        l1, l2 = (cp_locs if cp_locs else (-0.5, 0.5))
        k_time = _TimeOnly(_changepoint_time_kernel(l1, l2))
    elif name == "changepoint_1":                     # 2-segment
        l1 = (cp_locs[0] if cp_locs else 0.0)
        k_time = _TimeOnly(_changepoint_time_kernel_1(l1))
    elif name == "matern32":
        k_time = _time_kernel("matern32")
    else:
        k_time = _time_kernel(name)

    k_wave = Matern32(active_dims=[1])

    if not free_lambda:
        gpflow.set_trainable(k_wave.lengthscales, False)

    return k_time * k_wave