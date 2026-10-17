"""Cross-checks between noise.py/allan.py and the INDEPENDENT validation package.

WHY THIS FILE EXISTS. validate_model.py generates a trace with noise.py and
recovers its parameters with allan.py. Both are the same author's code, so a
CONSISTENT convention error passes that check perfectly and leaves the model
wrong against the datasheet: if the generator and the estimator share the same
mistaken definition of noise density, the round trip is clean and the physics
is not. That is not hypothetical -- it is exactly the class of bug raised as
finding 2 in docs/validation_findings_for_model.md (one-sided vs two-sided
density, the factor of sqrt(2)), and it was caught by a human reading the
datasheet, not by any test here.

imu_diff_sim.validation is an independently written package, already
cross-checked against allantools, so pairing it with ours breaks the circle in
both directions:

  A. generate with OURS, estimate with THEIRS  -> catches allan.py conventions
  B. generate with THEIRS, estimate with OURS  -> catches noise.py conventions

Direction B is the one that matters most, because noise.py is what feeds the
simulator and the search.

The mutation tests at the bottom plant the two classic convention bugs in
noise.py and assert the cross-check FAILS. The validation package's own
mutation tests plant bugs in its generator, not ours. A check that has never
been observed failing has not been shown to have any power.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from imu_diff_sim import allan, noise
from imu_diff_sim.params import ImuParams
from imu_diff_sim.validation import adev_jax, synth

DT = 0.005          # 200 Hz, the planned sim IMU rate
N_SAMPLES = 2_000_000
TRUE_N = 1.116e-4   # rad/s/sqrt(Hz)
TRUE_K = 1.06e-5    # rad/s^2/sqrt(Hz)

#: Single-realization recovery scatter is ~11% for K (see validate_model.py),
#: so these tolerances are set by sampling noise, not by expected bias.
TOL_N = 0.06
TOL_K = 0.30


def _our_params(scale_white: float = 1.0, scale_rw: float = 1.0) -> ImuParams:
    """Gyro-only parameters; accelerometer silenced so the channels are clean."""
    return ImuParams(
        gyroscope_noise_density=jnp.asarray(TRUE_N * scale_white),
        accelerometer_noise_density=jnp.asarray(0.0),
        gyroscope_random_walk=jnp.asarray(TRUE_K * scale_rw),
        accelerometer_random_walk=jnp.asarray(0.0),
        gyroscope_scale_factor=jnp.asarray(0.0),
        accelerometer_scale_factor=jnp.asarray(0.0),
        gyroscope_turn_on_bias=jnp.zeros(3),
        accelerometer_turn_on_bias=jnp.zeros(3),
    )


def _our_trace(params: ImuParams, seed: int = 0, n: int = N_SAMPLES) -> np.ndarray:
    eps = noise.draw_noise(jax.random.PRNGKey(seed), (n,))
    _, out = noise.corrupt_trace(
        noise.initial_bias(params), noise.zero_trace(n), params, eps, DT
    )
    return np.asarray(out.gyroscope)[:, 0]


def _their_fit(rate: np.ndarray) -> tuple[float, float]:
    """Estimate and fit with the independent package only."""
    taus, adev, _ = adev_jax.overlapping_adev(rate, DT)
    coeffs, _ = adev_jax.fit_nbk(
        taus, adev, DT, len(rate), terms=("N", "K")
    )
    return float(coeffs.N), float(coeffs.K)


# --------------------------------------------------------------------------
# A. our generator, their estimator
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_our_trace_recovered_by_independent_estimator():
    """noise.py -> validation.adev_jax. Nothing of ours in the estimation path."""
    n_hat, k_hat = _their_fit(_our_trace(_our_params()))
    assert n_hat == pytest.approx(TRUE_N, rel=TOL_N), (
        f"independent estimator recovers N={n_hat:.4g} from our generator, "
        f"input was {TRUE_N:.4g} (ratio {n_hat/TRUE_N:.3f}). A ratio near "
        "sqrt(2) or 1/sqrt(2) means a one-sided/two-sided density mismatch."
    )
    assert k_hat == pytest.approx(TRUE_K, rel=TOL_K), (
        f"independent estimator recovers K={k_hat:.4g}, input {TRUE_K:.4g} "
        f"(ratio {k_hat/TRUE_K:.3f}). A ratio near sqrt(3) means a dropped /3."
    )


# --------------------------------------------------------------------------
# B. their generator, our estimator  -- the direction that guards noise.py
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_independent_trace_recovered_by_our_estimator():
    """validation.synth -> allan.py. Nothing of ours in the generation path.

    Together with A, this pins both conventions against an outside reference
    rather than against each other.
    """
    rate = np.asarray(
        synth.synth_trace(
            jax.random.PRNGKey(1), N_SAMPLES, DT,
            synth.NoiseCoeffs(N=TRUE_N, B=0.0, K=TRUE_K),
        )
    )
    taus, adev, _ = allan.overlapping_allan_deviation(rate, DT)
    n_hat = allan.fit_white_noise(taus, adev)
    k_hat = allan.fit_random_walk(taus, adev)
    assert n_hat == pytest.approx(TRUE_N, rel=TOL_N), (
        f"our estimator recovers N={n_hat:.4g} from an independent generator, "
        f"input {TRUE_N:.4g} (ratio {n_hat/TRUE_N:.3f})"
    )
    assert k_hat == pytest.approx(TRUE_K, rel=TOL_K), (
        f"our estimator recovers K={k_hat:.4g}, input {TRUE_K:.4g} "
        f"(ratio {k_hat/TRUE_K:.3f})"
    )


@pytest.mark.slow
def test_both_generators_agree_on_the_same_coefficients():
    """Our trace and theirs must look statistically identical to one estimator.

    The sharpest form of the check: if the two generators disagree, exactly
    one of them has the convention wrong, and neither round-trip test on its
    own would reveal it.
    """
    ours = _their_fit(_our_trace(_our_params(), seed=2))
    theirs = _their_fit(np.asarray(
        synth.synth_trace(jax.random.PRNGKey(2), N_SAMPLES, DT,
                          synth.NoiseCoeffs(N=TRUE_N, B=0.0, K=TRUE_K))
    ))
    assert ours[0] == pytest.approx(theirs[0], rel=0.08), (
        f"generators disagree on N: ours {ours[0]:.4g} vs theirs {theirs[0]:.4g} "
        f"(ratio {ours[0]/theirs[0]:.4f})"
    )
    assert ours[1] == pytest.approx(theirs[1], rel=0.35), (
        f"generators disagree on K: ours {ours[1]:.4g} vs theirs {theirs[1]:.4g} "
        f"(ratio {ours[1]/theirs[1]:.4f})"
    )


# --------------------------------------------------------------------------
# Mutation tests: plant the bug in OUR generator, the cross-check must catch it
# --------------------------------------------------------------------------

def _mutated_step(factor_white: float = 1.0, factor_rw: float = 1.0):
    """noise.step with a deliberate scaling bug in one term."""
    real = noise.step

    def mutant(bias, true_imu, params, eps, dt):
        bad = params._replace(
            gyroscope_noise_density=params.gyroscope_noise_density * factor_white,
            gyroscope_random_walk=params.gyroscope_random_walk * factor_rw,
        )
        return real(bias, true_imu, bad, eps, dt)

    return mutant


@pytest.mark.slow
@pytest.mark.parametrize(
    "factor_white,factor_rw,label",
    [
        (np.sqrt(2.0), 1.0, "sqrt(2) in the white-noise density"),
        (1.0, np.sqrt(3.0), "dropped /3 in the random-walk term"),
    ],
)
def test_mutation_is_caught_by_the_independent_estimator(
    monkeypatch, factor_white, factor_rw, label
):
    """Plant a classic convention bug; the cross-check must fail.

    These are the two errors that survive a self-consistent round trip,
    because a generator and estimator sharing the mistake agree perfectly.
    """
    monkeypatch.setattr(noise, "step", _mutated_step(factor_white, factor_rw))
    n_hat, k_hat = _their_fit(_our_trace(_our_params(), seed=3))

    n_ok = abs(n_hat / TRUE_N - 1.0) <= TOL_N
    k_ok = abs(k_hat / TRUE_K - 1.0) <= TOL_K
    assert not (n_ok and k_ok), (
        f"MUTATION NOT CAUGHT ({label}): independent estimator still recovered "
        f"N={n_hat:.4g} (true {TRUE_N:.4g}) and K={k_hat:.4g} (true {TRUE_K:.4g}) "
        "within tolerance. The cross-check has no power against this bug."
    )
