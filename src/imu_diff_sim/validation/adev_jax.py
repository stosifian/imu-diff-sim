"""Overlapping Allan deviation, confidence bounds and an N/B/K fit, in JAX.

WHY A SECOND IMPLEMENTATION. imu_diff_sim.allan is NumPy by design, and that
is the right call for the main pipeline. This module exists to be checked
AGAINST it (and against allantools): two independent implementations that
agree are evidence; one implementation that agrees with itself is not. JAX
earns its place here for one reason -- the parametric bootstrap below needs
hundreds of synthesize-estimate-refit cycles, and those vmap/lax.map cleanly.

FLOAT64. Every public function runs under `jax.enable_x64(True)` locally,
rather than flipping the global flag, so importing this module does not change
the dtype of anything in the noise model. Without float64 the cumulative sum
loses the long-tau end of the curve to cancellation.

CONFIDENCE BOUNDS, two different things:

  per-point   chi-squared interval on each sigma(tau), with equivalent degrees
              of freedom from the Howe/Allan/Barnes formulas (NIST SP 1065,
              sec. 5.3). These are for plots and for weighting the fit.
  per-fit     parametric bootstrap on the fitted (N, B, K): resynthesize from
              the fit, re-estimate, refit, take percentiles. This is what the
              protocol's pass/fail thresholds use, because adjacent overlapping
              ADEV points are strongly correlated and a weighted least-squares
              covariance that ignores that is overconfident.

Both are conditional on the three-term model being the right model. A thermal
ramp or a dropped-sample glitch is outside it, and no interval here covers it.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .synth import NoiseCoeffs, synth_trace

#: sigma_flicker(tau) = B * sqrt(2 ln2 / pi), IEEE 952 Annex C.
FLICKER_FACTOR_SQ = 2.0 * np.log(2.0) / np.pi

# Noise-type codes for the EDF formulas. Rate-noise names, with the
# frequency-stability equivalents (what SP 1065 calls them) alongside.
WHITE, FLICKER, RANDOM_WALK = 0, 1, 2  # white FM, flicker FM, random-walk FM


def _x64(fn):
    """Run `fn` with 64-bit JAX types, without touching the global flag."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def default_ms(n: int, points_per_decade: int = 10, max_frac: float = 0.1) -> np.ndarray:
    """Log-spaced unique integer cluster sizes from 1 to max_frac * n.

    The default cap of n/10 keeps at least ~10 independent clusters at the
    longest tau. Past that the per-point EDF drops below ~5 and the point
    carries little information but a lot of leverage in a log-log fit.
    """
    m_max = max(1, int(max_frac * n))
    n_pts = int(np.ceil(np.log10(m_max) * points_per_decade)) + 1
    return np.unique(np.round(np.logspace(0, np.log10(m_max), n_pts)).astype(np.int64))


def _phase(rate: Array, dt: float) -> Array:
    """theta = [0, dt*cumsum(rate - mean)]: n+1 phase points."""
    x = rate - jnp.mean(rate)
    return jnp.concatenate([jnp.zeros(1, x.dtype), jnp.cumsum(x) * dt])


def _avar_at(theta: Array, m: Array, dt: float) -> tuple[Array, Array]:
    """Overlapping Allan variance at one (possibly traced) cluster size m.

    Indices are gathered and masked rather than sliced, so m can be a traced
    value and one compiled kernel serves every tau. Terms are k = 0 .. M-2m-1
    with M = n+1 phase points.
    """
    M = theta.shape[0]
    k = jnp.arange(M)
    n_terms = M - 2 * m
    a = theta[jnp.minimum(k + 2 * m, M - 1)]
    b = theta[jnp.minimum(k + m, M - 1)]
    d = jnp.where(k < n_terms, a - 2.0 * b + theta, 0.0)
    tau = m * dt
    return jnp.sum(d * d) / (2.0 * tau * tau * n_terms), n_terms


def _adev_core(rate: Array, dt: float, ms: Array) -> tuple[Array, Array]:
    theta = _phase(rate, dt)
    avar, n_terms = jax.lax.map(lambda m: _avar_at(theta, m, dt), ms)
    return jnp.sqrt(avar), n_terms


@_x64
def overlapping_adev(
    rate: np.ndarray, dt: float, ms: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a 1-D rate signal.

    Args:
        rate: (n,) rate samples [X].
        dt:   sample interval [s].
        ms:   integer cluster sizes; default `default_ms(n)`. Values outside
              1 <= m <= n/2 are dropped, not silently evaluated.

    Returns:
        (taus, adev, n_terms) as NumPy float64 / int arrays. n_terms is the
        number of OVERLAPPING second differences summed (n + 1 - 2m), which is
        what the EDF formulas need -- they account for the overlap themselves.
    """
    rate = np.asarray(rate, dtype=np.float64)
    if rate.ndim != 1:
        raise ValueError(f"rate must be 1-D, got shape {rate.shape}")
    n = rate.shape[0]
    ms = default_ms(n) if ms is None else np.unique(np.asarray(ms, dtype=np.int64))
    ms = ms[(ms >= 1) & (2 * ms <= n)]
    if ms.size == 0:
        raise ValueError("no valid cluster sizes for this record length")
    adev, n_terms = jax.jit(_adev_core)(jnp.asarray(rate), dt, jnp.asarray(ms))
    return ms * dt, np.asarray(adev), np.asarray(n_terms)


# ---------------------------------------------------------------------------
# Per-point confidence: EDF + chi-squared
# ---------------------------------------------------------------------------

def edf_oadev(n_phase: Array, m: Array, noise_type: Array) -> Array:
    """Equivalent degrees of freedom of the overlapping Allan variance.

    Howe, Allan & Barnes (1981) approximations as tabulated in NIST SP 1065
    (Riley 2008) sec. 5.3, with N = number of PHASE points (n_samples + 1):

      white FM:    [3(N-1)/(2m) - 2(N-2)/N] * 4m^2 / (4m^2 + 5)
      flicker FM:  m == 1: 2(N-2)^2 / (2.3N - 4.9);  m >= 2: 5N^2 / (4m(N+3m))
      RW FM:       (N-2)/m * ((N-1)^2 - 3m(N-1) + 4m^2) / (N-3)^2

    These are approximations, and they are TESTED rather than trusted:
    selftest check C5 compares 2/edf against the empirical scatter of
    sigma^2 across seeds. Clamped to [1, N - 2m].
    """
    N = jnp.asarray(n_phase, dtype=jnp.result_type(float))
    m = jnp.asarray(m, dtype=N.dtype)
    white = (3.0 * (N - 1.0) / (2.0 * m) - 2.0 * (N - 2.0) / N) * 4.0 * m**2 / (4.0 * m**2 + 5.0)
    flicker = jnp.where(
        m == 1, 2.0 * (N - 2.0) ** 2 / (2.3 * N - 4.9), 5.0 * N**2 / (4.0 * m * (N + 3.0 * m))
    )
    rw = (N - 2.0) / m * ((N - 1.0) ** 2 - 3.0 * m * (N - 1.0) + 4.0 * m**2) / (N - 3.0) ** 2
    edf = jnp.select([noise_type == WHITE, noise_type == FLICKER], [white, flicker], rw)
    return jnp.clip(edf, 1.0, N - 2.0 * m)


def chi2_ppf(p: Array, k: Array, newton_steps: int = 30) -> Array:
    """Chi-squared quantile: Wilson-Hilferty start, Newton-polished on the CDF.

    Jittable stand-in for scipy.stats.chi2.ppf. Wilson-Hilferty alone is NOT
    good enough here: its lower-tail quantile is 16% low at k = 3 and collapses
    to 0 at k = 1, and the lower-tail quantile sets the UPPER error bar at the
    longest taus -- exactly where EDF is smallest and the bar matters most.
    Newton steps in log(q) on jax.scipy.stats.chi2.cdf fix that (tested
    against scipy to 1e-6 for k in [1, 1e5]).
    """
    p = jnp.asarray(p, dtype=jnp.result_type(float))
    k = jnp.asarray(k, dtype=p.dtype)
    z = jax.scipy.special.ndtri(p)
    c = 2.0 / (9.0 * k)
    q0 = k * jnp.maximum(1.0 - c + z * jnp.sqrt(c), 1e-3) ** 3
    chi2 = jax.scipy.stats.chi2

    def newton(_, logq):
        q = jnp.exp(logq)
        step = (chi2.cdf(q, k) - p) / jnp.maximum(chi2.pdf(q, k) * q, 1e-300)
        return logq - jnp.clip(step, -2.0, 2.0)

    return jnp.exp(jax.lax.fori_loop(0, newton_steps, newton, jnp.log(q0)))


@_x64
def adev_ci(adev: np.ndarray, edf: np.ndarray, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided chi-squared interval on each sigma(tau): (lower, upper)."""
    adev, edf = jnp.asarray(adev), jnp.asarray(edf)
    a = (1.0 - conf) / 2.0
    lo = adev * jnp.sqrt(edf / chi2_ppf(1.0 - a, edf))
    hi = adev * jnp.sqrt(edf / chi2_ppf(a, edf))
    return np.asarray(lo), np.asarray(hi)


# ---------------------------------------------------------------------------
# Fit: sigma^2(tau) = N^2/tau + (2 ln2/pi) B^2 + K^2 tau/3
# ---------------------------------------------------------------------------

def _design(taus: Array) -> Array:
    """Columns for coefficients (N^2, B^2, K^2). The model is LINEAR in them."""
    return jnp.stack(
        [1.0 / taus, jnp.full_like(taus, FLICKER_FACTOR_SQ), taus / 3.0], axis=1
    )


# All 8 supports of a 3-coefficient NNLS problem; row s marks the free terms.
_SUPPORTS = jnp.array([[(s >> i) & 1 for i in range(3)] for s in range(8)], dtype=bool)


def _nnls3(A: Array, y: Array, w: Array, allowed: Array) -> Array:
    """Exact weighted non-negative least squares for 3 unknowns.

    The NNLS optimum is the unconstrained LS solution on its own support, so
    enumerating all 8 supports, discarding infeasible ones, and keeping the
    lowest loss is exact -- and, unlike a log-parameterized optimizer, it can
    return B^2 = 0 exactly. That matters: the null test (selftest C4) asks
    whether the fit can say "there is no flicker term", and a fit that
    parameterizes B as exp(.) can only ever say "very small".
    """
    sw = jnp.sqrt(w)
    Aw, yw = A * sw[:, None], y * sw

    def solve(support):
        support = support & allowed
        As = Aw * support[None, :]
        G = As.T @ As + jnp.diag(~support * 1.0)
        x = jnp.linalg.solve(G, As.T @ yw) * support
        loss = jnp.sum((yw - Aw @ x) ** 2)
        feasible = jnp.all(x >= 0.0)
        return x, jnp.where(feasible, loss, jnp.inf)

    xs, losses = jax.vmap(solve)(_SUPPORTS)
    return xs[jnp.argmin(losses)]


def _fit_core(
    taus: Array, adev: Array, ms: Array, n_phase: Array, allowed: Array, iters: int = 20
) -> tuple[Array, Array]:
    """Iteratively reweighted NNLS in the variance domain.

    Var(sigma_hat^2) ~= 2 sigma^4 / edf, so the weight on each point is
    edf / (2 v^2) with v the CURRENT model variance (not the noisy estimate:
    weighting by the estimate biases the fit low, because points that happen
    to fall low get more weight). EDF depends on which noise type dominates at
    that tau, so it is recomputed from the current fit each iteration.
    """
    A = _design(taus)
    y = adev**2

    def body(_, x):
        parts = A * x[None, :]
        v = jnp.maximum(parts.sum(axis=1), 1e-300)
        dominant = jnp.argmax(parts, axis=1)
        edf = edf_oadev(n_phase, ms, dominant)
        return _nnls3(A, y, edf / (2.0 * v**2), allowed)

    x0 = _nnls3(A, y, 1.0 / y**2, allowed)
    x = jax.lax.fori_loop(0, iters, body, x0)
    dominant = jnp.argmax(A * x[None, :], axis=1)
    return jnp.sqrt(x), edf_oadev(n_phase, ms, dominant)


class AdevFit(NamedTuple):
    coeffs: NoiseCoeffs          # point estimate
    lower: NoiseCoeffs           # bootstrap lower bound (conf level below)
    upper: NoiseCoeffs           # bootstrap upper bound
    conf: float
    taus: np.ndarray             # taus used in the fit (window applied)
    adev: np.ndarray
    edf: np.ndarray              # per-point EDF under the fitted model
    boot: np.ndarray             # (R, 3) bootstrap replicates of (N, B, K)


def _window(taus, tau_window):
    lo, hi = tau_window
    return (taus >= lo) & (taus <= hi)


@_x64
def fit_nbk(
    taus: np.ndarray,
    adev: np.ndarray,
    dt: float,
    n_samples: int,
    tau_window: tuple[float, float] = (0.0, np.inf),
    terms: tuple[str, ...] = ("N", "B", "K"),
) -> tuple[NoiseCoeffs, np.ndarray]:
    """Point fit of (N, B, K) over an EXPLICIT tau window.

    The window is an argument, not something detected from the curve: an
    automatic slope detector picks a different window for every log and makes
    results irreproducible. `terms` drops coefficients from the model entirely
    (fixed at zero), e.g. ("N", "K") for a model with no flicker term.

    Returns (coeffs, per-point edf).
    """
    keep = _window(np.asarray(taus), tau_window)
    taus_k = jnp.asarray(np.asarray(taus)[keep])
    ms = jnp.round(taus_k / dt)
    allowed = jnp.array([t in terms for t in ("N", "B", "K")])
    c, edf = jax.jit(_fit_core)(
        taus_k, jnp.asarray(np.asarray(adev)[keep]), ms, float(n_samples + 1), allowed
    )
    return NoiseCoeffs(*(float(v) for v in c)), np.asarray(edf)


@functools.lru_cache(maxsize=16)
def _compiled(n, dt, ms, keep, allowed, batch_size):
    """Jitted (estimate, bootstrap) pair, cached on everything that fixes
    shapes, so repeated fits of same-length records (coverage test, several
    axes of one log) compile once. Must be called under enable_x64."""
    ms_j = jnp.asarray(ms)
    keep_j = jnp.asarray(keep)
    ms_fit = ms_j[keep_j].astype(jnp.float64)
    taus_fit = ms_fit * dt
    allowed_j = jnp.asarray(allowed)
    n_phase = float(n + 1)

    def estimate(x):
        adev, _ = _adev_core(x, dt, ms_j)
        return _fit_core(taus_fit, adev[keep_j], ms_fit, n_phase, allowed_j)

    def bootstrap(c_hat, keys):
        def one_replicate(key):
            x = synth_trace(key, n, dt, NoiseCoeffs(c_hat[0], c_hat[1], c_hat[2]))
            return estimate(x)[0]

        return jax.lax.map(one_replicate, keys, batch_size=batch_size)

    return jax.jit(estimate), jax.jit(bootstrap)


@_x64
def fit_with_ci(
    rate: np.ndarray,
    dt: float,
    ms: np.ndarray | None = None,
    tau_window: tuple[float, float] = (0.0, np.inf),
    terms: tuple[str, ...] = ("N", "B", "K"),
    n_boot: int = 200,
    conf: float = 0.95,
    seed: int = 0,
    batch_size: int = 8,
) -> AdevFit:
    """ADEV -> (N, B, K) fit -> parametric-bootstrap confidence interval.

    Bootstrap: synthesize `n_boot` traces of the same length from the FITTED
    coefficients (validation.synth, not noise.py), run the identical ADEV and
    fit on each, and take the central `conf` percentile interval of each
    coefficient. Replicates are processed with lax.map in batches of
    `batch_size` (vmapped inside a batch) to bound memory at large n.

    What the interval does NOT cover: model misspecification (thermal ramps,
    a GM bump, sample drops) and non-stationarity. It answers "how much would
    this fit move on another record from the same three-term process", not
    "is this the right process".
    """
    rate = np.asarray(rate, dtype=np.float64)
    n = rate.shape[0]
    ms = default_ms(n) if ms is None else np.unique(np.asarray(ms, dtype=np.int64))
    ms = ms[(ms >= 1) & (2 * ms <= n)]
    taus_all = ms * dt
    keep = _window(taus_all, tau_window)
    if keep.sum() < 3:
        raise ValueError("fewer than 3 taus inside tau_window")

    allowed = tuple(t in terms for t in ("N", "B", "K"))
    estimate, bootstrap = _compiled(n, float(dt), tuple(ms), tuple(keep), allowed, batch_size)

    adev_all, _ = jax.jit(_adev_core)(jnp.asarray(rate), dt, jnp.asarray(ms))
    c_hat, edf = estimate(jnp.asarray(rate))
    coeffs = NoiseCoeffs(*(float(v) for v in c_hat))
    boot = bootstrap(c_hat, jax.random.split(jax.random.PRNGKey(seed), n_boot))
    boot = np.asarray(boot)

    a = 100.0 * (1.0 - conf) / 2.0
    lo, hi = np.percentile(boot, [a, 100.0 - a], axis=0)
    return AdevFit(
        coeffs=coeffs,
        lower=NoiseCoeffs(*map(float, lo)),
        upper=NoiseCoeffs(*map(float, hi)),
        conf=conf,
        taus=taus_all[keep],
        adev=np.asarray(adev_all)[keep],
        edf=np.asarray(edf),
        boot=boot,
    )
