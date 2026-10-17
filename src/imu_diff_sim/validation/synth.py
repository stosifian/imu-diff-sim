"""Independent synthetic rate-noise generator for the pipeline self-test.

INDEPENDENCE RULE. Nothing in this package imports `imu_diff_sim.noise` or
`imu_diff_sim.allan`. The self-test exists to catch convention errors that the
model and its own Allan code could share (a sqrt(2), a sqrt(dt), the /3 in the
random walk). A generator that reused noise.py would inherit exactly those
errors and the round trip would still close. This one is written from the
IEEE 952 Annex C power spectral densities, and the generator is itself checked
against those spectra with scipy's Welch estimator (selftest check G1), so it
does not lean on any ADEV code to prove itself right.

PSD CONVENTION, pinned here because it is where a factor of sqrt(2) hides.
IEEE 952 quotes TWO-SIDED rate PSDs. The one-sided PSD that scipy.signal.welch
returns, and that a datasheet "noise density" is usually quoted as, is twice
that:

    term            two-sided S(f)          one-sided           ADEV sigma^2(tau)
    white (N)       N^2                     2 N^2               N^2 / tau
    flicker (B)     B^2 / (2 pi f)          B^2 / (pi f)        (2 ln2 / pi) B^2
    random walk (K) K^2 / (2 pi f)^2        2 K^2 / (2 pi f)^2  K^2 tau / 3

So N is a TWO-sided density. A one-sided ASD of a [rad/s/sqrt(Hz)] means
N = a / sqrt(2). Which convention the BMI270 datasheet uses is a question for
the datasheet, not for this code.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class NoiseCoeffs(NamedTuple):
    """IEEE 952 rate-noise coefficients, SI units of the rate signal X."""

    N: float  # white,        [X / sqrt(Hz)] two-sided == [X * sqrt(s)]
    B: float  # flicker,      [X]
    K: float  # random walk,  [X / s / sqrt(Hz)]


def white(key: Array, n: int, N: float, dt: float) -> Array:
    """White rate noise: per-sample std N / sqrt(dt) (variance N^2 / dt)."""
    return (N / jnp.sqrt(dt)) * jax.random.normal(key, (n,))


def random_walk(key: Array, n: int, K: float, dt: float) -> Array:
    """Rate random walk: increments of std K * sqrt(dt), cumulative sum.

    Var after time T is K^2 T, independent of dt -- the defining property.
    """
    return jnp.cumsum(K * jnp.sqrt(dt) * jax.random.normal(key, (n,)))


def flicker_kasdin(key: Array, n: int, B: float, dt: float) -> Array:
    """1/f rate noise by Kasdin & Walter (1992, IEEE FCS) fractional filtering.

    White noise of per-sample variance Q is convolved with the impulse
    response of (1 - z^-1)^(-alpha/2), alpha = 1:

        h_0 = 1,   h_k = h_{k-1} * (k - 1 + alpha/2) / k

    The result has two-sided PSD  Q dt / |2 sin(pi f dt)|^alpha, which for
    f << 1/dt is Q / (2 pi f). Matching IEEE 952's B^2 / (2 pi f) gives
    Q = B^2 -- the per-sample input std is B, with no dt factor. That absence
    is not an oversight; it is what makes flicker noise scale-invariant.

    The convolution is linear (zero-padded FFT of length 2n), not circular, so
    the record start is a genuine start rather than a wrap-around of its end.
    """
    alpha = 1.0
    k = jnp.arange(1, n, dtype=jnp.result_type(float))
    h = jnp.concatenate([jnp.ones(1), jnp.cumprod((k - 1.0 + alpha / 2.0) / k)])
    w = B * jax.random.normal(key, (n,))
    nfft = 2 * n
    y = jnp.fft.irfft(jnp.fft.rfft(w, nfft) * jnp.fft.rfft(h, nfft), nfft)
    return y[:n]


def quantize(x: Array, lsb: float) -> Array:
    """Round to the ADC step. Adds ~lsb^2/12 white variance per sample."""
    return jnp.round(x / lsb) * lsb


def synth_trace(
    key: Array,
    n: int,
    dt: float,
    coeffs: NoiseCoeffs,
    lsb: float | None = None,
) -> Array:
    """Sum of independent white + flicker + random-walk rate noise.

    Any coefficient may be zero; that term is then exactly absent, which is
    what the null tests rely on. Three independent keys, one per term.
    """
    k_w, k_b, k_k = jax.random.split(key, 3)
    x = white(k_w, n, coeffs.N, dt)
    x = x + flicker_kasdin(k_b, n, coeffs.B, dt)
    x = x + random_walk(k_k, n, coeffs.K, dt)
    if lsb is not None:
        x = quantize(x, lsb)
    return x
