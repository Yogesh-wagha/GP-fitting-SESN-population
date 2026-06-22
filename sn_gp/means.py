"""
means.py  --  per-band mean functions for the MOGP.

"Per-band" means each band gets its OWN parameters (g and r peak at different
times with different amplitudes). We implement that by giving each mean a band
index per row and gathering the right parameters. Input X has columns
[time, wavelength]; run.py also passes a band-index array the mean uses.

To keep gpflow's MeanFunction signature (which only sees X), we map each row's
wavelength to its band index internally from a fixed list of band wavelengths.

--mean_func names: constant, polynomial, bazin
"""

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.functions import MeanFunction
from gpflow import Parameter


def _band_index_from_wave(Xwave, band_waves):
    """Map each row's (standardized) wavelength to its band index 0..B-1 by
       nearest match. band_waves is the standardized wavelength of each band."""
    bw = tf.constant(band_waves, dtype=Xwave.dtype)               # [B]
    d = tf.abs(Xwave[:, None] - bw[None, :])                      # [N, B]
    return tf.argmin(d, axis=1)                                   # [N]


class ConstantPerBand(MeanFunction):
    """m_b = c_b  (one constant per band)."""
    def __init__(self, n_bands, band_waves):
        super().__init__()
        self.band_waves = band_waves
        self.c = Parameter(np.zeros(n_bands))

    def __call__(self, X):
        idx = _band_index_from_wave(X[:, 1], self.band_waves)
        return tf.gather(self.c, idx)[:, None]


class PolynomialPerBand(MeanFunction):
    """m_b(t) = m_b*t + b_b + alpha_b * exp(-(t-l_b)^2 / (2 sigma_b^2)).
       Linear trend + a Gaussian bump (corrected: NEGATIVE exponent)."""
    def __init__(self, n_bands, band_waves):
        super().__init__()
        self.band_waves = band_waves
        self.slope = Parameter(np.zeros(n_bands))
        self.inter = Parameter(np.zeros(n_bands))
        self.alpha = Parameter(np.ones(n_bands))
        self.loc   = Parameter(np.zeros(n_bands))
        self.sigma = Parameter(np.ones(n_bands), transform=gpflow.utilities.positive())

    def __call__(self, X):
        t = X[:, 0]
        idx = _band_index_from_wave(X[:, 1], self.band_waves)
        m = tf.gather(self.slope, idx)
        b = tf.gather(self.inter, idx)
        a = tf.gather(self.alpha, idx)
        l = tf.gather(self.loc, idx)
        s = tf.gather(self.sigma, idx)
        val = m * t + b + a * tf.exp(-0.5 * ((t - l) / s) ** 2)
        return val[:, None]


class BazinPerBand(MeanFunction):
    """m_b(t) = A_b * exp(-(t-t0_b)/tau_tail_b) / (1 + exp(-(t-t0_b)/tau_rise_b)) + c_b.
       Standard Bazin SN light-curve form, one parameter set per band."""
    def __init__(self, n_bands, band_waves):
        super().__init__()
        self.band_waves = band_waves
        pos = gpflow.utilities.positive()
        self.A        = Parameter(np.ones(n_bands),  transform=pos)
        self.t0       = Parameter(np.zeros(n_bands))
        self.tau_rise = Parameter(np.ones(n_bands),  transform=pos)
        self.tau_tail = Parameter(np.ones(n_bands) * 5.0, transform=pos)
        self.c        = Parameter(np.zeros(n_bands))

    def __call__(self, X):
        t = X[:, 0]
        idx = _band_index_from_wave(X[:, 1], self.band_waves)
        A  = tf.gather(self.A, idx)
        t0 = tf.gather(self.t0, idx)
        tr = tf.gather(self.tau_rise, idx)
        tt = tf.gather(self.tau_tail, idx)
        c  = tf.gather(self.c, idx)
        dt = t - t0
        val = A * tf.exp(-dt / tt) / (1.0 + tf.exp(-dt / tr)) + c
        return val[:, None]


def build_mean(name, n_bands, band_waves):
    if name == "constant":
        return ConstantPerBand(n_bands, band_waves)
    if name == "polynomial":
        return PolynomialPerBand(n_bands, band_waves)
    if name == "bazin":
        return BazinPerBand(n_bands, band_waves)
    raise ValueError(f"unknown mean_func '{name}'")