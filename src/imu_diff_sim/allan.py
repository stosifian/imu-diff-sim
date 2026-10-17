"""Overlapping Allan deviation and noise-coefficient fitting.

CONCEPT NOTE.

This module is deliberately NumPy, not JAX. It is offline analysis -- nothing
here ever needs a gradient or a GPU, and JAX's tracing would only make the
debugging harder. Not everything in a JAX project should be JAX. The rule this
project uses: JAX for anything inside a gradient or a rollout, NumPy for
anything that consumes a finished trace.

WHAT THIS FILE IS FOR. It is not decoration. It is the test that tells you
whether noise.py is correct, because the Allan deviation of a simulated trace
has a known analytic form in terms of the parameters you fed in:

    white noise (ARW/VRW)      sigma(tau) = N / sqrt(tau)          slope -1/2
    bias instability (flicker) sigma(tau) = B * sqrt(2*ln2/pi)     slope  0
                                          ~ 0.664 * B
    random walk (RRW/ARW_a)    sigma(tau) = K * sqrt(tau / 3)      slope +1/2

and the coefficients map directly onto the Kalibr parameters, because the units
are the same thing written differently:

    rad/s/sqrt(Hz)   ==  rad/sqrt(s)     so  N == gyroscope_noise_density
    rad/s^2/sqrt(Hz) ==  rad/s^1.5       so  K == gyroscope_random_walk

VERIFY THOSE UNIT IDENTITIES YOURSELF before trusting the fit -- they are the
reason this works at all, and they take thirty seconds on paper.

The single most useful consequence: sigma(tau = 1 s) equals N directly. If you
generate a trace with a known noise density and read the Allan deviation at one
second, you should get that density back. If you get it back scaled by exactly
sqrt(dt) or 1/sqrt(dt), the bug is in noise.py step(), not here -- the two
sqrt(dt) factors are the only places that scaling can come from. That is a
much sharper diagnostic than "the curve looks wrong".
"""

from __future__ import annotations

import numpy as np


def overlapping_allan_deviation(
    rate: np.ndarray,
    dt: float,
    taus: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal.

    Args:
        rate:  (N,) or (N, K) sampled rate signal -- gyro in rad/s, accel in
               m/s^2. Multi-axis input is processed per column.
        dt:    sampling interval [s].
        taus:  averaging times [s]. If None, `default_taus` is used.

    Returns:
        (taus_actual, adev, n_clusters)

        `taus_actual` is not `taus`: cluster size m must be an integer, so the
        achievable averaging times are m*dt and requested values get snapped.
        Returning the snapped values matters -- regressing a slope against the
        requested x values instead of the achieved ones is a silent error.

        `adev` has shape (T,) for 1-D input, (T, K) for 2-D.

        `n_clusters` is the number of INDEPENDENT clusters, N // m -- not the
        N - 2m + 1 overlapping terms actually summed. The confidence formula
        wants independent clusters, and reporting the overlapping count would
        make the error bars look far better than they are.

    Estimator, on the integrated signal theta = dt * cumsum(rate):

        sigma^2(tau) = 1/(2 tau^2 (N - 2m + 1))
                       * sum_k (theta[k+2m] - 2 theta[k+m] + theta[k])^2

    The overlapping form is used rather than the non-overlapping one because
    at long tau a non-overlapping estimator has only a handful of clusters and
    the curve degenerates into noise precisely where the random-walk slope is
    read.
    """
    rate = np.asarray(rate, dtype=np.float64)
    if rate.ndim == 1:
        single = True
        rate = rate[:, None]
    elif rate.ndim == 2:
        single = False
    else:
        raise ValueError(f"rate must be 1-D or 2-D, got shape {rate.shape}")

    n = rate.shape[0]
    if taus is None:
        taus = default_taus(n, dt)
    taus = np.asarray(taus, dtype=np.float64)

    # Snap requested taus onto integer cluster sizes, drop duplicates and any
    # m for which the sum below would have no terms.
    m = np.unique(np.round(taus / dt).astype(np.int64))
    m = m[(m >= 1) & (2 * m <= n)]
    if m.size == 0:
        raise ValueError("no valid cluster sizes; record is too short for these taus")
    taus_actual = m * dt

    # Mean removal before integration. Allan variance is blind to a constant
    # rate offset by construction (the second difference of a linear theta is
    # zero), so this changes nothing physically -- it only stops theta growing
    # without bound, which is what protects the long-tau end from cancellation.
    centred = rate - rate.mean(axis=0, keepdims=True)
    theta = np.concatenate(
        [np.zeros((1, centred.shape[1])), np.cumsum(centred, axis=0) * dt], axis=0
    )

    adev = np.empty((m.size, rate.shape[1]), dtype=np.float64)
    for i, mi in enumerate(m):
        # Vectorised second difference; a Python loop over k would be minutes.
        d = theta[2 * mi :] - 2.0 * theta[mi : n + 1 - mi] + theta[: n + 1 - 2 * mi]
        tau = mi * dt
        avar = np.sum(d * d, axis=0) / (2.0 * tau * tau * d.shape[0])
        adev[i] = np.sqrt(avar)

    n_clusters = np.maximum(n // m, 1)
    if single:
        adev = adev[:, 0]
    return taus_actual, adev, n_clusters


def default_taus(n_samples: int, dt: float, points_per_decade: int = 10) -> np.ndarray:
    """Log-spaced averaging times, from dt up to n_samples*dt/10.

    Log spacing is not cosmetic: the coefficient fits are straight lines in
    log-log, so linear spacing would concentrate all the leverage on the
    longest -- and noisiest -- averaging times.

    The upper limit is N/10, not N/2. At m = N/2 the estimator has a single
    term and is pure noise; capping at ten independent clusters keeps every
    returned point meaningful. This is more conservative than some references
    and it matters, because the random-walk fit lives at the long-tau end.
    """
    m_max = max(int(n_samples // 10), 1)
    n_points = max(int(points_per_decade * np.log10(max(m_max, 2))), 2)
    m = np.unique(np.round(np.logspace(0.0, np.log10(m_max), n_points)).astype(np.int64))
    return m * dt


def allan_confidence(adev: np.ndarray, n_clusters: np.ndarray) -> np.ndarray:
    """1-sigma uncertainty on each Allan deviation point.

    Standard approximation: fractional uncertainty 1/sqrt(2(n_clusters - 1)).

    OPTIMISTIC, and the write-up should say so. The overlapping estimator's
    clusters are correlated, so the true uncertainty is larger than this;
    `n_clusters` is already the independent count rather than the overlapping
    term count, which removes the worst of the overstatement but not all of it.

    These bars are not decoration. The long-tau end of an Allan curve is far
    less certain than the short-tau end, and a plot without them invites a
    reader to over-read a random-walk slope fitted to three effectively
    independent points.
    """
    adev = np.asarray(adev, dtype=np.float64)
    n_clusters = np.asarray(n_clusters, dtype=np.float64)
    denom = np.sqrt(2.0 * np.maximum(n_clusters - 1.0, 1e-12))
    frac = 1.0 / denom
    if adev.ndim == 2:
        frac = frac[:, None]
    return adev * frac


def _forced_slope_fit(
    taus: np.ndarray,
    adev: np.ndarray,
    window: tuple[float, float],
    slope: float,
    tau_ref: float,
) -> float:
    """Fit adev = C * (tau/tau_ref)**slope over `window`, return C.

    The slope is FORCED rather than fitted. Fitting it would let a contaminated
    window quietly return a coefficient from a region that is not the regime it
    claims to be -- the forced fit at least fails visibly when the window is
    wrong, because the residual is large.
    """
    lo, hi = window
    keep = (taus >= lo) & (taus <= hi) & np.isfinite(adev) & (adev > 0)
    if keep.sum() < 2:
        raise ValueError(
            f"fit window [{lo:.4g}, {hi:.4g}] s contains {int(keep.sum())} usable "
            f"points; available taus span [{taus.min():.4g}, {taus.max():.4g}] s"
        )
    log_c = np.mean(np.log(adev[keep]) - slope * np.log(taus[keep] / tau_ref))
    return float(np.exp(log_c))


def fit_white_noise(
    taus: np.ndarray,
    adev: np.ndarray,
    window: tuple[float, float] | None = None,
) -> float:
    """Recover N [X/sqrt(s)] from the slope -1/2 region: sigma(tau) = N/sqrt(tau).

    N equals the Kalibr noise density, because rad/s/sqrt(Hz) and rad/sqrt(s)
    are the same units written differently. Equivalently N = sigma(tau = 1 s).

    `window` defaults to [10, 100] x the shortest available tau -- one decade,
    starting clear of the very shortest averaging times where quantization and
    any anti-alias filtering distort real data. The window is an explicit
    argument because an automatic slope-detector that silently picks a
    different region per device is how a result becomes irreproducible; record
    whatever window you use.
    """
    taus = np.asarray(taus, dtype=np.float64)
    if window is None:
        window = (10.0 * taus.min(), 100.0 * taus.min())
    return _forced_slope_fit(taus, np.asarray(adev), window, -0.5, 1.0)


def fit_random_walk(
    taus: np.ndarray,
    adev: np.ndarray,
    window: tuple[float, float] | None = None,
) -> float:
    """Recover K [X/s/sqrt(s)] from the slope +1/2 region.

    sigma(tau) = K * sqrt(tau/3). The division by 3 is real and is the most
    commonly dropped constant in this subject: a recovered K that is high by
    sqrt(3) ~ 1.73 means it was forgotten.

    `window` defaults to the longest available decade. This regime needs long
    tau, which needs a long record -- an 8-hour log supports tau to roughly
    1000 s. A window running past that is fitting noise, which is why
    `default_taus` caps at ten independent clusters.
    """
    taus = np.asarray(taus, dtype=np.float64)
    if window is None:
        window = (taus.max() / 10.0, taus.max())
    return _forced_slope_fit(taus, np.asarray(adev), window, 0.5, 3.0)


def fit_bias_instability(
    taus: np.ndarray,
    adev: np.ndarray,
    flatness_tol: float = 0.1,
    require_flat: bool = True,
) -> float:
    """Recover B from the flat minimum: B = min(adev) / 0.664.

    Returns NaN when the curve is not actually flat at its minimum, and that is
    the important behaviour rather than a nuisance. Any curve with white noise
    at short tau and random walk at long tau HAS a minimum whether or not
    flicker noise exists. Taking argmin unconditionally always returns a
    number, and that number is meaningless unless the curve is genuinely flat
    there -- it is just the white/random-walk crossover.

    This connects to a modelling choice in noise.py: a pure random walk, which
    is what this project's bias model uses, contains NO flicker term, so a
    synthetic trace from it should return NaN here. If it does not, something
    is wrong. Real data from a MEMS part generally does show a flicker floor,
    which is one of the honest gaps between our model and a real device.

    `flatness_tol` is the maximum |local log-log slope| at the minimum that
    still counts as flat. Set `require_flat=False` to get the raw argmin value
    anyway, for diagnostics.
    """
    taus = np.asarray(taus, dtype=np.float64)
    adev = np.asarray(adev, dtype=np.float64)
    i = int(np.nanargmin(adev))
    value = float(adev[i] / 0.664)
    if not require_flat:
        return value
    if i == 0 or i == len(adev) - 1:
        return float("nan")  # minimum at an endpoint: no floor, just the edge
    slope = float(
        (np.log(adev[i + 1]) - np.log(adev[i - 1]))
        / (np.log(taus[i + 1]) - np.log(taus[i - 1]))
    )
    return value if abs(slope) <= flatness_tol else float("nan")
