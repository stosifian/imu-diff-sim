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

    # --- Turn-on bias: constant for the episode, SEARCHED not sampled ---
    # Per-axis, shape (3,). Deterministic parameters rather than random draws,
    # so the adversarial search can target them directly instead of seeing
    # them as nuisance randomness. One searched constant also stands in for
    # the slow thermal offset, which over a 20 s episode is indistinguishable
    # from a constant.
    gyroscope_turn_on_bias: Array       # [rad / s],   shape (3,)
    accelerometer_turn_on_bias: Array   # [m / s^2],   shape (3,)


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
    "gyroscope_turn_on_bias": 0.0,
    "accelerometer_turn_on_bias": 0.0,
}

BMI270_PROVENANCE: dict[str, str] = {
    "gyroscope_noise_density": (
        "COMMUNITY 1.116e-4 rad/s/sqrt(Hz), used as nominal. DATASHEET "
        "(DS000-08 Tables 10-12, fig 12) implies N = 8.64e-5 in PERFORMANCE "
        "mode and 1.23e-4 in NORMAL mode; the datasheet density is one-sided, "
        "with rms/(density*sqrt(f_3dB)) ~ 1.0 at every ODR <= 1.6 kHz, so "
        "there is no sqrt(2) convention factor to apply. The Luxonis nominal "
        "matches NORMAL mode, not performance mode. THE SIM MUST BE "
        "CONFIGURED FOR NORMAL MODE for this nominal to be self-consistent."
    ),
    "accelerometer_noise_density": (
        "COMMUNITY 1.405e-3 m/s^2/sqrt(Hz), used as nominal. DATASHEET implies "
        "1.11e-3; the measurement is 27% higher, which is unsurprising for a "
        "part inside a camera body. Datasheet accel noise varies by ODR/mode "
        "(Tables 8, 11, 12) -- revisit once the sim ODR is fixed."
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
        "Nominal 0.0 by construction: the box is centred on no scale error and "
        "the BOUND carries the physics. DATASHEET gyro sensitivity error "
        "+-2% without CRT (DS000-08 Table 3) -> GYRO_SCALE_FACTOR_BOUND."
    ),
    "accelerometer_scale_factor": (
        "Nominal 0.0 by construction. DATASHEET accel sensitivity tolerance "
        "+-0.4% (DS000-08 Table 2) -> ACCEL_SCALE_FACTOR_BOUND. Deliberately "
        "NOT the gyro's 2%: a shared bound would make the accel box 5x the "
        "datasheet spec and manufacture failures that no real part exhibits."
    ),
    "gyroscope_turn_on_bias": (
        "Nominal 0.0; the bound carries the physics. DATASHEET gyro zero-rate "
        "offset +-0.5 dps = +-8.727e-3 rad/s (DS000-08 Table 3). This is "
        "60-190x the within-episode random-walk drift at T_ep = 20 s, which "
        "makes it the dominant bias term by a wide margin and the reason "
        "initial_bias() no longer returns zero."
    ),
    "accelerometer_turn_on_bias": (
        "Nominal 0.0; the bound carries the physics. DATASHEET accel zero-g "
        "offset +-20 mg = +-0.19613 m/s^2 (DS000-08 Table 2)."
    ),
    "GYRO_SCALE_FACTOR_BOUND": (
        "DATASHEET +-2%, the gyro sensitivity error WITHOUT CRT (DS000-08 "
        "Table 3). A device whose integrator never ran component retrimming "
        "is a real device and belongs at the box edge. Sensitivity "
        "temperature coefficient stacks on top of this; not modelled."
    ),
    "ACCEL_SCALE_FACTOR_BOUND": (
        "DATASHEET +-0.4%, accel sensitivity tolerance (DS000-08 Table 2)."
    ),
    "GYRO_TURN_ON_BIAS_BOUND": (
        "DATASHEET +-0.5 dps zero-rate offset (DS000-08 Table 3), converted "
        "to rad/s. Add +-TCO*dT (0.015 dps/K) if the robot's thermal range "
        "exceeds the lab's; not currently included."
    ),
    "ACCEL_TURN_ON_BIAS_BOUND": (
        "DATASHEET +-20 mg zero-g offset (DS000-08 Table 2), converted to "
        "m/s^2. Add +-TCO*dT (0.25 mg/K) for wider thermal range; not "
        "currently included."
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
    # (lo, hi) multiplicative factors on nominal, each tagged with its source.
    "gyroscope_noise_density": (0.5, 5.0),      # LITERATURE (Suvorkin 2024)
    "accelerometer_noise_density": (0.5, 5.0),  # LITERATURE (Suvorkin 2024)
    "gyroscope_random_walk": (0.5, 20.0),       # UNSOURCED -- see below
    "accelerometer_random_walk": (0.5, 20.0),   # UNSOURCED -- see below
}

#: Provenance per box edge. The random-walk box is deliberately wider because
#: the datasheet specifies no random walk at all and the only measurement is
#: the single Luxonis device whose author doubted it -- but "wider because we
#: know less" is a judgement, not a source, and the write-up must say so.
SEARCH_BOX_PROVENANCE: dict[str, str] = {
    "gyroscope_noise_density": (
        "LITERATURE: spans roughly the consumer-MEMS grade spread in Suvorkin "
        "et al. 2024 (10.3390/s24061953). The datasheet performance-mode "
        "value (8.64e-5) and normal-mode value (1.23e-4) both lie inside."
    ),
    "accelerometer_noise_density": (
        "LITERATURE: as above. Datasheet-implied 1.11e-3 lies inside."
    ),
    "gyroscope_random_walk": (
        "UNSOURCED. No datasheet value exists; the box is widened to (0.5, 20) "
        "to reflect that we know less, not because a source supports that "
        "range. Report any failure found here at a lower plausibility tier."
    ),
    "accelerometer_random_walk": "UNSOURCED, as above.",
}

# ---------------------------------------------------------------------------
# Bounded (tanh-squashed) parameters. Every bound below is a datasheet number;
# see BMI270_PROVENANCE for the table reference. Bounds are PER SENSOR because
# the gyro and accelerometer specs differ by 5x, and a shared bound would
# manufacture failures no real part exhibits.
# ---------------------------------------------------------------------------

_DPS_TO_RAD_S = math.pi / 180.0
_MG_TO_M_S2 = 9.80665e-3

GYRO_SCALE_FACTOR_BOUND: float = 0.02            # +-2%, no-CRT (Table 3)
ACCEL_SCALE_FACTOR_BOUND: float = 0.004          # +-0.4%     (Table 2)
GYRO_TURN_ON_BIAS_BOUND: float = 0.5 * _DPS_TO_RAD_S    # +-0.5 dps  (Table 3)
ACCEL_TURN_ON_BIAS_BOUND: float = 20.0 * _MG_TO_M_S2    # +-20 mg    (Table 2)

#: field -> (bound, shape). Scalars have shape (), per-axis fields (3,).
BOUNDED_FIELDS: dict[str, tuple[float, tuple[int, ...]]] = {
    "gyroscope_scale_factor": (GYRO_SCALE_FACTOR_BOUND, ()),
    "accelerometer_scale_factor": (ACCEL_SCALE_FACTOR_BOUND, ()),
    "gyroscope_turn_on_bias": (GYRO_TURN_ON_BIAS_BOUND, (3,)),
    "accelerometer_turn_on_bias": (ACCEL_TURN_ON_BIAS_BOUND, (3,)),
}

#: Field order of the flat unconstrained vector: log-squashed fields first
#: (in SEARCH_BOX_FACTORS order), then the bounded fields, each contributing
#: prod(shape) or 1 entries.
LOG_FIELDS: tuple[str, ...] = tuple(SEARCH_BOX_FACTORS)
BOUNDED_FIELD_NAMES: tuple[str, ...] = tuple(BOUNDED_FIELDS)
UNCONSTRAINED_FIELDS: tuple[str, ...] = LOG_FIELDS + BOUNDED_FIELD_NAMES


def unconstrained_size() -> int:
    """Length of the flat search vector."""
    n = len(LOG_FIELDS)
    for _, shape in BOUNDED_FIELDS.values():
        n += int(math.prod(shape)) if shape else 1
    return n


def _log_bounds(name: str) -> tuple[float, float]:
    """Natural-log bounds of the search interval for a log-squashed field.

    Uses math.log, not jnp.log, deliberately. These bounds are compile-time
    constants built from Python floats. Inside a jit trace every jnp operation
    is staged out, so jnp.log would return a tracer and float() on it raises
    ConcretizationTypeError -- which breaks every jitted call site, including
    the entire search arm. Keep arithmetic on static configuration in plain
    Python.
    """
    lo, hi = SEARCH_BOX_FACTORS[name]
    nominal = _NOMINAL[name]
    return math.log(lo * nominal), math.log(hi * nominal)


def to_unconstrained(params: ImuParams) -> Array:
    """Map physical parameters to an unbounded vector the optimizer walks on.

    Inverse of `from_unconstrained`; a test enforces the round trip.

    A parameter sitting exactly ON a box edge maps to +-inf, since
    arctanh(+-1) is infinite. Nominal is strictly inside every box: the factor
    ranges all bracket 1.0, and every bounded field has nominal 0.0.
    """
    values = []
    for name in LOG_FIELDS:
        lo_log, hi_log = _log_bounds(name)
        t = 2.0 * (jnp.log(getattr(params, name)) - lo_log) / (hi_log - lo_log) - 1.0
        values.append(jnp.atleast_1d(jnp.arctanh(t)))
    for name in BOUNDED_FIELD_NAMES:
        bound, shape = BOUNDED_FIELDS[name]
        v = jnp.broadcast_to(jnp.asarray(getattr(params, name)), shape or ())
        values.append(jnp.atleast_1d(jnp.arctanh(v / bound)))
    return jnp.concatenate(values)


def from_unconstrained(u: Array) -> ImuParams:
    """Inverse of `to_unconstrained`. This is where physical validity lives.

    Noise densities and random walks: tanh (to bound) composed with log (to
    span decades),

        param = exp( lo_log + (hi_log - lo_log) * (tanh(u) + 1) / 2 )

    so u -> -inf gives lo*nominal, u -> +inf gives hi*nominal, and u = 0 gives
    the GEOMETRIC mean sqrt(lo*hi)*nominal -- not nominal, unless lo*hi == 1.

    Scale factors and turn-on biases: tanh alone, in linear space, because
    those errors are signed; log would be wrong. Each uses its own per-sensor
    datasheet bound.

    Because tanh saturates, box edges are reachable only asymptotically and
    the gradient there goes to zero. That is usually desirable -- the
    optimizer decelerates into the wall rather than slamming into it -- but a
    failure mode sitting at an edge shows up as the search stalling near it,
    which signals the box is too narrow.
    """
    out: dict[str, Array] = {}
    i = 0
    for name in LOG_FIELDS:
        lo_log, hi_log = _log_bounds(name)
        out[name] = jnp.exp(
            lo_log + (hi_log - lo_log) * (jnp.tanh(u[i]) + 1.0) / 2.0
        )
        i += 1
    for name in BOUNDED_FIELD_NAMES:
        bound, shape = BOUNDED_FIELDS[name]
        n = int(math.prod(shape)) if shape else 1
        chunk = u[i : i + n]
        out[name] = bound * jnp.tanh(chunk.reshape(shape) if shape else chunk[0])
        i += n
    return ImuParams(**out)
