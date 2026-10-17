"""Tests for the differentiable IMU noise model.

The statistical tests here are the ones that catch a wrong sqrt(dt). Read the
docstrings before implementing -- each one states the physical invariant it is
protecting, and those invariants are the actual specification of noise.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from imu_diff_sim import noise
from imu_diff_sim.params import ImuParams, bmi270_nominal, from_unconstrained, to_unconstrained


def _params(**overrides) -> ImuParams:
    """BMI270 nominal with fields overridden, as jnp scalars."""
    p = bmi270_nominal()
    return p._replace(**{k: jnp.asarray(v) for k, v in overrides.items()})


def _run(params, n, dt, seed=0):
    eps = noise.draw_noise(jax.random.PRNGKey(seed), (n,))
    bias0 = noise.initial_bias(params)
    return noise.corrupt_trace(bias0, noise.zero_trace(n), params, eps, dt)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_shapes_and_finiteness():
    _, out = _run(bmi270_nominal(), n=1000, dt=0.005)
    assert out.gyroscope.shape == (1000, 3)
    assert out.accelerometer.shape == (1000, 3)
    assert jnp.all(jnp.isfinite(out.gyroscope))
    assert jnp.all(jnp.isfinite(out.accelerometer))


def test_deterministic_given_eps():
    """Same eps, same params -> bit-identical output.

    This is the reparameterisation trick stated as a test. If this fails there
    is a jax.random call inside the model, and no gradient below is trustworthy.
    """
    a = _run(bmi270_nominal(), n=500, dt=0.005, seed=3)[1]
    b = _run(bmi270_nominal(), n=500, dt=0.005, seed=3)[1]
    np.testing.assert_array_equal(np.asarray(a.gyroscope), np.asarray(b.gyroscope))


# --------------------------------------------------------------------------
# Physics: the sqrt(dt) scalings
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.02, 0.005, 0.001])
def test_white_noise_sample_std_scales_as_inverse_sqrt_dt(dt):
    """Discrete white-noise std must be density / sqrt(dt).

    Physical statement: a noise density is a per-root-Hz quantity. Sampling
    faster does not reduce the noise in each sample -- it increases it, because
    each sample averages over a shorter window. Sampling 4x faster raises the
    per-sample std by 2x.

    If your implementation has this inverted, this test fails in the obvious
    direction and you have found the bug in ten seconds rather than in week 3.
    """
    nd = 1e-3
    p = _params(gyroscope_noise_density=nd, gyroscope_random_walk=0.0,
                gyroscope_scale_factor=0.0)
    _, out = _run(p, n=200_000, dt=dt)
    expected = nd / np.sqrt(dt)
    measured = float(np.std(np.asarray(out.gyroscope)))
    assert measured == pytest.approx(expected, rel=0.02), (
        f"dt={dt}: expected per-sample std {expected:.4g}, measured {measured:.4g}, "
        f"ratio {measured/expected:.4g} (a ratio of dt or 1/dt means sqrt(dt) is "
        "in the wrong place; a ratio of sqrt(dt)^2 means it was applied twice)"
    )


@pytest.mark.parametrize("dt", [0.02, 0.005])
def test_bias_random_walk_grows_as_sqrt_of_elapsed_time(dt):
    """Bias std after T seconds must be random_walk * sqrt(T), independent of dt.

    This is the invariant that pins the OTHER sqrt(dt), and it goes the
    opposite way to the white-noise one. Variance of a random walk accumulates
    linearly in elapsed time, so the per-step increment must shrink as dt
    shrinks -- otherwise a finer timestep would make the device worse, which is
    physically absurd.

    Measured across independent seeds, not across time within one trace: the
    bias at the end of a single walk is one sample from the distribution, not
    an estimate of its width.
    """
    T = 100.0
    rw = 1e-4
    n = int(T / dt)
    p = _params(gyroscope_noise_density=0.0, gyroscope_random_walk=rw)

    finals = []
    for seed in range(200):
        final_bias, _ = _run(p, n=n, dt=dt, seed=seed)
        finals.append(np.asarray(final_bias.gyroscope))
    measured = float(np.std(np.stack(finals)))
    expected = rw * np.sqrt(T)
    assert measured == pytest.approx(expected, rel=0.15), (
        f"dt={dt}: bias std after {T}s was {measured:.4g}, expected {expected:.4g}. "
        "This must not depend on dt."
    )


def test_scale_factor_is_multiplicative_on_signal_plus_bias():
    """A scale error multiplies the measurement, so it needs a signal to act on.

    Run a non-zero true input: with zero noise and zero bias, a +1% scale error
    must produce exactly 1.01x the input. This catches the model applying the
    scale factor to the noise as well, or applying it additively.
    """
    p = _params(gyroscope_noise_density=0.0, gyroscope_random_walk=0.0,
                gyroscope_scale_factor=0.01)
    n = 10
    true = noise.ImuSample(gyroscope=jnp.ones((n, 3)), accelerometer=jnp.zeros((n, 3)))
    eps = noise.draw_noise(jax.random.PRNGKey(0), (n,))
    _, out = noise.corrupt_trace(noise.initial_bias(p), true, p, eps, 0.005)
    np.testing.assert_allclose(np.asarray(out.gyroscope), 1.01, rtol=1e-6)


def test_gyro_and_accel_noise_are_independent():
    """Four noise sources, four keys. Correlated channels mean a key was reused."""
    p = _params(gyroscope_random_walk=0.0, accelerometer_random_walk=0.0)
    _, out = _run(p, n=50_000, dt=0.005)
    g = np.asarray(out.gyroscope)[:, 0]
    a = np.asarray(out.accelerometer)[:, 0]
    corr = float(np.corrcoef(g, a)[0, 1])
    assert abs(corr) < 0.02, f"gyro and accel white noise are correlated (r={corr:.3f})"


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------

def test_gradient_is_finite_and_nonzero_for_every_parameter():
    """Every parameter must actually influence the loss.

    A zero gradient for some field means that field is unused -- silently
    dropped in `step`, most likely -- and the search would then explore a
    dimension that does nothing.
    """
    eps = noise.draw_noise(jax.random.PRNGKey(0), (256,))
    true = noise.ImuSample(gyroscope=jnp.ones((256, 3)), accelerometer=jnp.ones((256, 3)))

    def loss(u):
        p = from_unconstrained(u)
        _, out = noise.corrupt_trace(noise.initial_bias(p), true, p, eps, 0.005)
        return jnp.mean(out.gyroscope ** 2) + jnp.mean(out.accelerometer ** 2)

    u = to_unconstrained(bmi270_nominal())
    g = jax.grad(loss)(u)
    assert jnp.all(jnp.isfinite(g)), f"non-finite gradient: {g}"
    assert jnp.all(jnp.abs(g) > 0), f"some parameter has zero gradient: {g}"


def test_gradient_matches_central_finite_differences():
    """The 10-minute test the brief (5) calls out. Run it before trusting anything.

    Central differences, relative step, on the unconstrained vector -- which is
    the space the optimizer actually moves in, so this validates the transform
    and the model together.
    """
    eps = noise.draw_noise(jax.random.PRNGKey(0), (128,))
    true = noise.ImuSample(gyroscope=jnp.ones((128, 3)), accelerometer=jnp.ones((128, 3)))

    def loss(u):
        p = from_unconstrained(u)
        _, out = noise.corrupt_trace(noise.initial_bias(p), true, p, eps, 0.005)
        return jnp.mean(out.gyroscope ** 2) + jnp.mean(out.accelerometer ** 2)

    u = jnp.asarray(to_unconstrained(bmi270_nominal()), dtype=jnp.float64) \
        if jax.config.jax_enable_x64 else to_unconstrained(bmi270_nominal())
    analytic = jax.grad(loss)(u)

    h = 1e-3
    for i in range(u.shape[0]):
        e = jnp.zeros_like(u).at[i].set(h)
        numeric = (loss(u + e) - loss(u - e)) / (2 * h)
        assert float(analytic[i]) == pytest.approx(float(numeric), rel=5e-2, abs=1e-8), (
            f"parameter {i}: analytic {float(analytic[i]):.6g} vs "
            f"numeric {float(numeric):.6g}"
        )
