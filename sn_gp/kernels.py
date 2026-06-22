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
                            RationalQuadratic, Constant, Linear, ChangePoints)


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


def _changepoint_time_kernel():
    """Non-stationary changepoint kernel on the time axis.

    SE-SNe adaptation of the SN II scheme (no hydrogen plateau):
      baseline (Constant) -> rise+decline (Matern32) -> Ni tail (Linear).
    Two changepoints whose locations are learned. NOTE: many parameters and
    fragile to optimize on sparse data -- here only so AIC/BIC can judge it.
    Locations are in STANDARDIZED time (data are standardized in run.py), so 0
    is roughly the mean epoch; init just inside that.
    """
    base = Constant(active_dims=[0])
    rise = Matern32(active_dims=[0])
    tail = Linear(active_dims=[0])
    return ChangePoints(
        kernels=[base, rise, tail],
        locations=[-0.5, 0.5],     # standardized-time guesses
        steepness=1.0,
    )


def build_kernel(name, free_lambda):
    """Return k_time * k_wave. If free_lambda is False, the wavelength length
       scale is fixed (sparse-band objects); otherwise it is optimized."""
    k_time = _time_kernel(name)
    k_wave = Matern32(active_dims=[1])

    if not free_lambda:
        # fix the wavelength length scale (value set/standardized in run.py,
        # which assigns k_wave.lengthscales before calling set_trainable)
        gpflow.set_trainable(k_wave.lengthscales, False)

    return k_time * k_wave