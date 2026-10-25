"""Tests for the physical <-> unconstrained parameter transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from imu_diff_sim.params import (
    BOUNDED_FIELDS,
    SEARCH_BOX_FACTORS,
    ImuParams,
    bmi270_nominal,
    from_unconstrained,
    to_unconstrained,
)


def test_round_trip_is_identity():
    """Physical -> unconstrained -> physical must return what went in.

    If this fails, every gradient in the project is being taken with respect to
    the wrong thing.
    """
    p = bmi270_nominal()
    back = from_unconstrained(to_unconstrained(p))
    for name, original in p._asdict().items():
        np.testing.assert_allclose(
            getattr(back, name), original, rtol=1e-6,
            err_msg=f"round trip changed {name}",
        )


@pytest.mark.parametrize("magnitude", [10.0, 50.0, 200.0])
def test_extreme_unconstrained_values_stay_physical(magnitude):
    """The transform, not a clip, is what enforces validity.

    An optimizer that takes a huge step must still land on a physically valid
    parameter set. Noise densities stay strictly positive; scale factors stay
    inside their bound. This is the property that lets the search run without
    any post-hoc clipping (which would be non-smooth and break gradients).
    """
    assert SEARCH_BOX_FACTORS, (
        "SEARCH_BOX_FACTORS is empty, so this test would pass vacuously. "
        "Fill it in before trusting a green run here."
    )
    u = to_unconstrained(bmi270_nominal())
    for sign in (+1.0, -1.0):
        p = from_unconstrained(u + sign * magnitude)
        nominal = bmi270_nominal()
        for name, (lo, hi) in SEARCH_BOX_FACTORS.items():
            value = float(getattr(p, name))
            base = float(getattr(nominal, name))
            assert value > 0, f"{name} left the positive reals"
            assert lo * base * (1 - 1e-6) <= value <= hi * base * (1 + 1e-6), (
                f"{name}={value:.4g} escaped its search box "
                f"[{lo*base:.4g}, {hi*base:.4g}] at u offset {sign*magnitude}. "
                "The transform, not a clip, must enforce this."
            )
        for name, (bound, _) in BOUNDED_FIELDS.items():
            v = np.abs(np.asarray(getattr(p, name)))
            assert np.all(v <= bound * (1 + 1e-6) + 1e-12), (
                f"{name} = {v} escaped its datasheet bound {bound} "
                f"at u offset {sign * magnitude}"
            )


def test_transform_is_differentiable():
    """jax.grad through the transform must be finite -- the search depends on it."""
    u = to_unconstrained(bmi270_nominal())
    g = jax.grad(lambda v: sum(jnp.sum(leaf) for leaf in from_unconstrained(v)))(u)
    assert jnp.all(jnp.isfinite(g)), "non-finite gradient through the parameter transform"


def test_log_space_equalises_gradient_scales():
    """The reason for log space, stated as a test.

    A 1% relative change in any parameter should move the unconstrained vector
    by roughly the same amount, regardless of whether the parameter is ~1e-5 or
    ~1e-3. Without log space, gyro random walk (1e-5) would be invisible next
    to accel noise density (1e-3) in the same gradient step.
    """
    p = bmi270_nominal()
    u0 = to_unconstrained(p)
    deltas = []
    for name in ("gyroscope_noise_density", "accelerometer_noise_density",
                 "gyroscope_random_walk", "accelerometer_random_walk"):
        bumped = p._replace(**{name: getattr(p, name) * 1.01})
        deltas.append(float(jnp.max(jnp.abs(to_unconstrained(bumped) - u0))))
    assert max(deltas) / min(deltas) < 2.0, (
        f"1% changes produce very different step sizes across parameters: {deltas}. "
        "Are the noise densities all in log space?"
    )


def test_every_parameter_has_provenance():
    """Provenance must cover every field, exactly.

    No code reads BMI270_PROVENANCE -- it exists for the write-up, which means
    nothing else would ever catch it going stale. This test is the only thing
    standing between you and a write-up sentence you cannot source. If you add a
    parameter, you document where its value came from.
    """
    from imu_diff_sim.params import BMI270_PROVENANCE

    # Module-level constants that are not ImuParams fields but still carry a
    # number someone has to defend in the write-up.
    extra_constants = {
        "GYRO_SCALE_FACTOR_BOUND", "ACCEL_SCALE_FACTOR_BOUND",
        "GYRO_TURN_ON_BIAS_BOUND", "ACCEL_TURN_ON_BIAS_BOUND",
    }

    fields = set(ImuParams._fields)
    documented = set(BMI270_PROVENANCE)
    assert fields <= documented, (
        f"undocumented parameters: {sorted(fields - documented)}"
    )
    stray = documented - fields - extra_constants
    assert not stray, f"provenance for non-existent parameters: {sorted(stray)}"
    assert extra_constants <= documented, (
        f"undocumented constants: {sorted(extra_constants - documented)}"
    )
    for name, note in BMI270_PROVENANCE.items():
        assert any(tag in note for tag in ("DATASHEET", "COMMUNITY", "MEASURED", "UNKNOWN")), (
            f"{name}: provenance note must start from a confidence tag "
            "(DATASHEET / COMMUNITY / MEASURED / UNKNOWN)"
        )


def test_transforms_work_under_jit():
    """Both transforms must survive jit -- the search arm jits everything.

    Regression test for a real bug: _log_bounds used float(jnp.log(...)), and
    inside a jit trace jnp operations are staged out, so float() raised
    ConcretizationTypeError. Config arithmetic must stay in plain Python.
    """
    import jax

    @jax.jit
    def round_trip(u):
        return to_unconstrained(from_unconstrained(u))

    u = to_unconstrained(bmi270_nominal())
    np.testing.assert_allclose(np.asarray(round_trip(u)), np.asarray(u), rtol=1e-6)

    @jax.jit
    def grad_through(u):
        return jax.grad(lambda v: sum(jnp.sum(leaf) for leaf in from_unconstrained(v)))(u)

    assert jnp.all(jnp.isfinite(grad_through(u)))


def test_every_search_box_edge_has_a_source():
    """Box edges define which failures count as real (brief 9), so each needs
    a source tag -- DATASHEET, LITERATURE, or an explicit UNSOURCED admission.
    An untagged edge is an unfalsifiable claim about what devices exist.
    """
    from imu_diff_sim.params import SEARCH_BOX_PROVENANCE

    assert set(SEARCH_BOX_PROVENANCE) == set(SEARCH_BOX_FACTORS), (
        "every search-box entry needs a provenance note: missing "
        f"{sorted(set(SEARCH_BOX_FACTORS) - set(SEARCH_BOX_PROVENANCE))}"
    )
    for name, note in SEARCH_BOX_PROVENANCE.items():
        assert any(t in note for t in ("DATASHEET", "LITERATURE", "MEASURED", "UNSOURCED")), (
            f"{name}: box provenance must carry a source tag"
        )
