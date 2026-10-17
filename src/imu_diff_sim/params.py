"""IMU noise parameters: physical values, provenance, and search-space transforms.

Naming follows the Kalibr convention (brief 4.1.1) so these parameters map onto
how the rest of the robotics community talks about IMU noise:
https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

Two spaces exist here:

  PHYSICAL      SI units, what a datasheet quotes, what Allan variance
                recovers. `ImuParams` holds these.
  UNCONSTRAINED What the optimizer searches over. Noise densities span ~1e-5
                to 1e-2, so they live in log space; everything is squashed
                through tanh so that box constraints are STRUCTURAL -- any
                real-valued vector maps to a physically valid parameter set,
                and the search never needs a clip (a clip is non-smooth and
                would break the gradients the whole project depends on).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class ImuParams(NamedTuple):
    """Physical IMU noise parameters, SI units, Kalibr naming.

    A NamedTuple is automatically a JAX pytree, so `jax.grad` of a function of
    an `ImuParams` returns an `ImuParams` of gradients with the same field
    names.

    Scalars are per-axis-identical. Extending a field to a length-3 array for
    per-axis values needs no change to noise.py -- broadcasting handles it.
    """

    # --- White noise (Allan deviation slope -1/2) ---
    gyroscope_noise_density: Array      # [rad / s / sqrt(Hz)]
    accelerometer_noise_density: Array  # [m / s^2 / sqrt(Hz)]

    # --- Bias diffusion (Allan deviation slope +1/2) ---
    gyroscope_random_walk: Array        # [rad / s^2 / sqrt(Hz)]
    accelerometer_random_walk: Array    # [m / s^3 / sqrt(Hz)]

    # --- Deterministic per-episode errors ---
    gyroscope_scale_factor: Array       # [dimensionless] fractional, 0.0 == perfect
    accelerometer_scale_factor: Array   # [dimensionless] fractional, 0.0 == perfect


# ---------------------------------------------------------------------------
# BMI270 reference values.
#
# Confidence tags, used by test_every_parameter_has_provenance:
#   DATASHEET  Bosch BST-BMI270-DS000, directly quoted.
#   COMMUNITY  Luxonis measurement of one BMI270 in an OAK camera
#              (github.com/luxonis/depthai-core issue #886): 10 h at 100 Hz,
#              ONE device, inside a camera body. The poster flagged the gyro
#              random walk as suspicious and was unsure the stream was raw.
#   MEASURED   Our own static bench log. Outranks COMMUNITY once it exists.
#   UNKNOWN    Not specified anywhere we have found.
# ---------------------------------------------------------------------------

_NOMINAL: dict[str, float] = {
    "gyroscope_noise_density": 1.116e-4,
    "accelerometer_noise_density": 1.405e-3,
    "gyroscope_random_walk": 1.06e-5,
    "accelerometer_random_walk": 7.27e-5,
    "gyroscope_scale_factor": 0.0,
    "accelerometer_scale_factor": 0.0,
}

BMI270_PROVENANCE: dict[str, str] = {
    "gyroscope_noise_density": (
        "COMMUNITY 1.116e-4 rad/s/sqrt(Hz), used as nominal. DATASHEET quotes "
        "typ. < 7 mdps/sqrt(Hz) in Performance mode -> 1.22e-4; the two do not "
        "conflict, since the datasheet figure is an upper bound on typical and "
        "the measurement sits under it. Nominal is the measurement because the "
        "search box centre should be a real device, not a spec ceiling; the "
        "datasheet value lies inside the box."
    ),
    "accelerometer_noise_density": (
        "COMMUNITY 1.405e-3 m/s^2/sqrt(Hz). DATASHEET cross-check: 160 ug/sqrt(Hz) "
        "at BW=100 Hz -> 1.57e-3 m/s^2/sqrt(Hz), consistent. Datasheet accel noise "
        "varies by ODR/mode (Tables 8, 11, 12) -- revisit once the sim ODR is fixed."
    ),
    "gyroscope_random_walk": (
        "COMMUNITY 1.06e-5 rad/s^2/sqrt(Hz). Not in the datasheet. WEAKEST NUMBER "
        "IN THIS FILE: the original poster flagged it as 'looking weird', possibly "
        "a CRT (component retrimming) artifact -- a retrim mid-log steps the gyro "
        "scale factor, which inflates apparent random walk at long tau. Search box "
        "is widened to (0.5, 20) to reflect the uncertainty."
    ),
    "accelerometer_random_walk": (
        "COMMUNITY 7.27e-5 m/s^3/sqrt(Hz). Not in the datasheet. Same single-device "
        "caveat; box widened to (0.5, 20)."
    ),
    "gyroscope_scale_factor": (
        "UNKNOWN as a nominal -- set to 0.0 (perfect sensor) so the search box is "
        "centred on no scale error. The BOUND, not the nominal, carries the physics; "
        "see SCALE_FACTOR_BOUND."
    ),
    "accelerometer_scale_factor": (
        "UNKNOWN as a nominal -- set to 0.0, as above."
    ),
    "SCALE_FACTOR_BOUND": (
        "UNKNOWN provenance: 0.02 was entered without a source. NEEDS A DATASHEET "
        "CITATION before the write-up. Intended basis (brief discussion): the "
        "WITHOUT-CRT gyro sensitivity-error spec, since a device whose integrator "
        "never ran component retrimming is a real device and belongs at the box "
        "edge. Note also that sensitivity temperature coefficient (TCS) stacks on "
        "top of the room-temperature tolerance, arguing for a wider bound."
    ),
}


def bmi270_nominal() -> ImuParams:
    """Nominal BMI270 parameters -- the reference point for the search box."""
    return ImuParams(**{k: jnp.asarray(v) for k, v in _NOMINAL.items()})


# ---------------------------------------------------------------------------
# Search box.
#
# Multiplicative factors on nominal, which is the natural scale for a log-space
# search and survives a change of nominal. Brief 9: a failure mode at an
# unphysical parameter value is not a finding, so these edges are an assertion
# that every device inside the box could plausibly ship.
#
# Asymmetry between the two groups is deliberate. Noise densities are anchored
# by a datasheet AND a measurement that agree, so (0.5, 5) spans roughly the
# consumer-MEMS grade spread in Suvorkin et al. 2024 (10.3390/s24061953).
# Random walks have NO datasheet value at all -- only the single doubted
# community measurement -- so the box is widened to (0.5, 20) to reflect that
# we know materially less about them. Documenting that asymmetry is the point.
#
# NOTE: scale factors are deliberately NOT in this dict. These are
# multiplicative factors on nominal, and scale-factor nominal is 0.0, so a
# factor box collapses to a point. Scale factor is signed and bounded, and is
# handled by SCALE_FACTOR_BOUND with a linear-space tanh instead.
# ---------------------------------------------------------------------------

SEARCH_BOX_FACTORS: dict[str, tuple[float, float]] = {
    "gyroscope_noise_density": (0.5, 5.0),
    "accelerometer_noise_density": (0.5, 5.0),
    "gyroscope_random_walk": (0.5, 20.0),
    "accelerometer_random_walk": (0.5, 20.0),
}

#: Max |fractional scale error|. See BMI270_PROVENANCE["SCALE_FACTOR_BOUND"] --
#: this value still needs a datasheet citation.
SCALE_FACTOR_BOUND: float = 0.02

#: Field order of the flat unconstrained vector. Log-squashed fields first (in
#: SEARCH_BOX_FACTORS order), then the linear-squashed scale factors.
LOG_FIELDS: tuple[str, ...] = tuple(SEARCH_BOX_FACTORS)
SCALE_FIELDS: tuple[str, ...] = (
    "gyroscope_scale_factor",
    "accelerometer_scale_factor",
)
UNCONSTRAINED_FIELDS: tuple[str, ...] = LOG_FIELDS + SCALE_FIELDS


def _log_bounds(name: str) -> tuple[float, float]:
    """Natural-log bounds of the search interval for a log-squashed field.

    Uses math.log, not jnp.log, deliberately. These bounds are compile-time
    constants built from Python floats. Inside a jit trace every jnp operation
    is staged out, so jnp.log would return a tracer and float() on it raises
    ConcretizationTypeError -- which breaks every jitted call site, including
    the entire week-3 search. Keep arithmetic on static configuration in plain
    Python.
    """
    lo, hi = SEARCH_BOX_FACTORS[name]
    nominal = _NOMINAL[name]
    return math.log(lo * nominal), math.log(hi * nominal)


def to_unconstrained(params: ImuParams) -> Array:
    """Map physical parameters to an unbounded vector the optimizer walks on.

    Inverse of `from_unconstrained`. Both must be written as a pair; a test
    enforces the round trip.

    Note that a parameter sitting exactly ON a box edge maps to +-inf, because
    arctanh(+-1) is infinite. Nominal must therefore be strictly inside the box
    (it is: every factor range brackets 1.0, and scale-factor nominal is 0.0,
    strictly inside +-SCALE_FACTOR_BOUND).
    """
    values = []
    for name in LOG_FIELDS:
        lo_log, hi_log = _log_bounds(name)
        # Invert: param = exp(lo + (hi-lo)*(tanh(u)+1)/2)
        t = 2.0 * (jnp.log(getattr(params, name)) - lo_log) / (hi_log - lo_log) - 1.0
        values.append(jnp.arctanh(t))
    for name in SCALE_FIELDS:
        values.append(jnp.arctanh(getattr(params, name) / SCALE_FACTOR_BOUND))
    return jnp.stack(values)


def from_unconstrained(u: Array) -> ImuParams:
    """Inverse of `to_unconstrained`. This is where physical validity lives.

    Noise densities and random walks: tanh (to bound) composed with log (to
    span decades),

        param = exp( lo_log + (hi_log - lo_log) * (tanh(u) + 1) / 2 )

    so u -> -inf gives lo*nominal, u -> +inf gives hi*nominal, and u = 0 gives
    the GEOMETRIC mean sqrt(lo*hi)*nominal -- not nominal, unless lo*hi == 1.
    Worth remembering when choosing where gradient-search restarts begin.

    Scale factors: tanh alone, in linear space, because a scale error is signed
    and log would be wrong.

    Because tanh saturates, parameters at the exact box edges are reachable
    only asymptotically and the gradient there goes to zero. That is usually
    desirable -- the optimizer decelerates into the wall rather than slamming
    into it -- but it means a failure mode sitting at the upper edge shows up
    as the search stalling near it, which is a signal the box is too narrow.
    """
    out: dict[str, Array] = {}
    for i, name in enumerate(LOG_FIELDS):
        lo_log, hi_log = _log_bounds(name)
        out[name] = jnp.exp(lo_log + (hi_log - lo_log) * (jnp.tanh(u[i]) + 1.0) / 2.0)
    for j, name in enumerate(SCALE_FIELDS, start=len(LOG_FIELDS)):
        out[name] = SCALE_FACTOR_BOUND * jnp.tanh(u[j])
    return ImuParams(**out)
