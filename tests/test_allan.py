"""Tests for the Allan deviation estimator.

These use synthetic signals with analytically known Allan deviations, so they
test the ESTIMATOR in isolation. The end-to-end test at the bottom then closes
the loop through noise.py -- that one is the credibility anchor in miniature.
"""

import jax
import numpy as np
import pytest

from imu_diff_sim import allan, noise
from imu_diff_sim.params import bmi270_nominal


def test_white_noise_gives_minus_half_slope():
    """Pure white noise: sigma(tau) = N / sqrt(tau), so log-log slope is -1/2."""
    rng = np.random.default_rng(0)
    dt, n, nd = 0.005, 500_000, 1e-3
    x = rng.normal(0.0, nd / np.sqrt(dt), n)

    taus, adev, _ = allan.overlapping_allan_deviation(x, dt)
    keep = (taus >= 10 * dt) & (taus <= 1.0)
    slope = np.polyfit(np.log(taus[keep]), np.log(adev[keep]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05), f"slope {slope:.3f}, expected -1/2"


def test_white_noise_density_recovered():
    """sigma(tau=1s) == N. The cleanest possible check on the estimator."""
    rng = np.random.default_rng(1)
    dt, n, nd = 0.005, 1_000_000, 1e-3
    x = rng.normal(0.0, nd / np.sqrt(dt), n)

    taus, adev, _ = allan.overlapping_allan_deviation(x, dt)
    assert allan.fit_white_noise(taus, adev) == pytest.approx(nd, rel=0.05)


def test_random_walk_gives_plus_half_slope_and_recovers_k():
    """Pure random walk: sigma(tau) = K * sqrt(tau/3), log-log slope +1/2.

    If the recovered K is off by sqrt(3) ~= 1.73, the /3 was dropped.
    """
    rng = np.random.default_rng(2)
    dt, n, K = 0.01, 500_000, 1e-3
    x = np.cumsum(rng.normal(0.0, K * np.sqrt(dt), n))

    taus, adev, _ = allan.overlapping_allan_deviation(x, dt)
    keep = (taus >= 10.0) & (taus <= 500.0)
    slope = np.polyfit(np.log(taus[keep]), np.log(adev[keep]), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.06), f"slope {slope:.3f}, expected +1/2"
    assert allan.fit_random_walk(taus, adev) == pytest.approx(K, rel=0.15)


def test_confidence_shrinks_with_cluster_count():
    """Error bars must widen at long tau, where there are fewer clusters."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1e-2, 200_000)
    taus, adev, n_clusters = allan.overlapping_allan_deviation(x, 0.005)
    conf = allan.allan_confidence(adev, n_clusters)
    frac = conf / adev
    assert frac[-1] > frac[0], "fractional uncertainty should grow with tau"


def test_taus_are_snapped_to_integer_clusters():
    """Requested taus are not necessarily achievable; returned ones must be exact."""
    taus, _, _ = allan.overlapping_allan_deviation(np.zeros(10_000) + 1e-9, 0.005)
    m = taus / 0.005
    np.testing.assert_allclose(m, np.round(m), atol=1e-9)


# --------------------------------------------------------------------------
# End to end: noise.py -> allan.py -> parameters back out
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_simulated_trace_recovers_its_own_parameters():
    """THE credibility anchor, in test form (brief 4.1, 9).

    Generate a long trace from known parameters, run the Allan estimator, and
    recover the parameters that went in. This is a self-consistency check, not
    a validation against reality -- it verifies the discretisation, the
    estimator, and the unit conventions all agree. Say exactly that in the
    write-up; do not call it validation.

    Failure signatures worth recognising:
      - recovered N off by sqrt(dt) or 1/sqrt(dt)  -> white-noise scaling in step()
      - recovered K off by sqrt(3)                 -> the /3 in fit_random_walk
      - recovered K off by dt or 1/dt              -> bias-walk scaling in step()
      - both off by the same factor                -> the estimator, not the model
    """
    dt, n = 0.005, 2_000_000
    p = bmi270_nominal()

    eps = noise.draw_noise(jax.random.PRNGKey(0), (n,))
    _, out = noise.corrupt_trace(
        noise.initial_bias(p), noise.zero_trace(n), p, eps, dt
    )
    gyro = np.asarray(out.gyroscope)[:, 0]

    taus, adev, _ = allan.overlapping_allan_deviation(gyro, dt)

    assert allan.fit_white_noise(taus, adev) == pytest.approx(
        float(p.gyroscope_noise_density), rel=0.10
    )
    assert allan.fit_random_walk(taus, adev) == pytest.approx(
        float(p.gyroscope_random_walk), rel=0.30
    )
