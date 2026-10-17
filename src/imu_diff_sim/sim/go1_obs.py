"""Go1 joystick observation layout and direct IMU error injection.

The `state` observation is 48-dimensional, built in
mujoco_playground/_src/locomotion/go1/joystick.py::_get_obs as:

    [ 0: 3]  local linear velocity   <- NOT directly measurable on a real
                                        robot; comes from a state estimator.
                                        Its presence means the policy has a
                                        partial route around IMU corruption.
    [ 3: 6]  gyro                    <- IMU, angular rate [rad/s]
    [ 6: 9]  projected gravity       <- IMU, accelerometer-derived unit vector
    [ 9:21]  joint angles - default pose (12)
    [21:33]  joint velocities (12)
    [33:45]  last action (12)
    [45:48]  velocity command (3)

INJECTION, NOT MODELLING. The functions here apply idealised error modes
directly, with no physics model and no datasheet. That is deliberate: the
point of the OFAT sweep is to find each channel's failure threshold first, and
only then ask whether a real device can reach it. Putting a sensor model in
the loop at this stage would confound "is the policy sensitive" with "is our
model right".

The env's own uniform DR (gyro +-0.2 rad/s, gravity +-0.05) must be disabled
for these numbers to mean anything -- otherwise every injection is measured on
top of a much larger noise floor. See ofat_sensitivity.py, which sets
noise_config.level = 0.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

GYRO = slice(3, 6)
GRAVITY = slice(6, 9)
LINVEL = slice(0, 3)
STATE_OBS_SIZE = 48

#: Reference magnitudes for the plots. See BMI270_PROVENANCE for sourcing.
BMI270_GYRO_WHITE_STD_50HZ = 1.116e-4 / (0.02 ** 0.5)   # 7.89e-4 rad/s
PLAYGROUND_GYRO_DR_STD = 0.2 / (3.0 ** 0.5)             # 0.1155 rad/s
PLAYGROUND_GRAVITY_DR_HALFWIDTH = 0.05

#: Datasheet turn-on bias bounds -- the reference lines that matter most, since
#: this is the only channel where a physically plausible value EXCEEDS what the
#: policy was trained against. Playground's DR is zero-mean i.i.d., so its
#: persistent component over an N-step episode is only DR_std/sqrt(N).
BMI270_GYRO_TURN_ON_BIAS_BOUND = 0.5 * 3.141592653589793 / 180.0   # 8.73e-3 rad/s
#: Tilt of the gravity unit vector from the accel zero-g OFFSET (+-20 mg).
#: NOT from the scale-factor spec: projected gravity is renormalised, so a
#: common-mode scale error changes the vector's length and not its
#: direction, producing zero tilt. Only an offset, or a per-axis
#: differential scale error, tilts it.
BMI270_ACCEL_TURN_ON_BIAS_TILT = 20e-3              # rad, approx
PLAYGROUND_GYRO_DR_EPISODE_MEAN_500 = PLAYGROUND_GYRO_DR_STD / (500 ** 0.5)


def inject_white(obs: Array, eps: Array, sigma: float, sl: slice = GYRO) -> Array:
    """Zero-mean i.i.d. noise, redrawn every step. Mimics the env's own DR.

    This is the channel a policy CAN filter: over an N-step episode its time
    average shrinks as sigma/sqrt(N).
    """
    return obs.at[sl].add(sigma * eps)


def inject_bias(obs: Array, bias: Array, sl: slice = GYRO) -> Array:
    """Constant offset, fixed for the whole episode.

    The channel a policy CANNOT filter: it does not average down, and it
    integrates into heading and attitude error. Absent from the env's DR,
    which is zero-mean i.i.d. per step.
    """
    return obs.at[sl].add(bias)


def inject_scale(obs: Array, scale_error: float, sl: slice = GYRO) -> Array:
    """Multiplicative sensitivity error, (1 + s) * signal.

    Structurally absent from the env's DR, which is purely additive. Note this
    scales the observation as the policy sees it, which for the gyro slice is
    the same as scaling the sensor.
    """
    return obs.at[sl].multiply(1.0 + scale_error)


def inject_gravity_tilt(obs: Array, tilt: Array) -> Array:
    """Constant tilt of the projected-gravity unit vector, renormalised.

    An accelerometer scale or bias error shows up as the robot believing
    "down" is in the wrong direction. Renormalising keeps it a unit vector,
    which is what the policy expects -- injecting an un-normalised vector
    would test a different and unphysical failure.
    """
    g = obs[GRAVITY] + tilt
    g = g / (jnp.linalg.norm(g) + 1e-9)
    return obs.at[GRAVITY].set(g)


def apply_latency(buffer: Array, obs_slice: Array) -> tuple[Array, Array]:
    """Push a new reading into a delay buffer, return (new_buffer, delayed).

    Latency is included because on real hardware it is frequently the dominant
    IMU-path error -- bus transfer, driver scheduling, and filter group delay
    stack up to several control periods -- and it is entirely absent from the
    env's DR. Cheap to test and worth ranking against the others.
    """
    return jnp.concatenate([buffer[1:], obs_slice[None, :]]), buffer[0]
