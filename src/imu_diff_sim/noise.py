"""Differentiable IMU noise model.

CONCEPT NOTE -- read before filling anything in.

There are two things this file has to get right, and they are independent.

(1) THE PHYSICS. The Kalibr measurement model, in continuous time:

        gyro:   w_meas(t) = (1 + s_g) * ( w_true(t) + b_g(t) ) + n_g(t)
        accel:  a_meas(t) = (1 + s_a) * ( a_true(t) + b_a(t) ) + n_a(t)

        b_dot(t) = n_b(t)                      (bias diffusion)

    where n_g, n_a, n_b are continuous-time white noise processes whose
    strengths are quoted as *densities* (per sqrt(Hz)), and s_g, s_a are the
    constant per-episode scale-factor errors.

    You are implementing the DISCRETE-TIME version of that at sampling interval
    dt, and the conversion is where the bugs live. The two sqrt(dt) factors go
    in OPPOSITE directions. Derive both before you write them -- do not look
    them up, because getting this wrong produces a model that still runs, still
    looks noisy, and is silently wrong by a factor that depends on your
    sampling rate.

    The derivation to do on paper:
      - White noise: the density sigma_nd has units [X/sqrt(Hz)]. Averaging
        white noise over a sample interval dt reduces its variance. What is the
        standard deviation of the discrete sample? Sanity check your answer by
        asking what happens as dt -> 0: should the per-sample noise grow or
        shrink?
      - Bias random walk: sigma_rw has units [X/s/sqrt(Hz)]. Over dt the bias
        accumulates a random increment. Variance of a random walk grows
        linearly in elapsed time. What is the standard deviation of one step?
        Sanity check: after total time T, the bias std should be
        sigma_rw * sqrt(T) regardless of how you chose dt. If your answer
        fails that test it is wrong.

    If the two factors come out the same way, you have made a mistake.

(2) THE DIFFERENTIABILITY. Every random number this model uses is drawn
    OUTSIDE it and passed in as `eps` -- standard normals, fixed. Inside, the
    parameters only ever multiply or add to those fixed draws. That is the
    reparameterisation trick, and it is the whole reason this project works:

        loss(params) = f(params, eps)     with eps held fixed

    is an ordinary deterministic function of params, so jax.grad differentiates
    it. If you called jax.random.normal(key, ...) *inside* here using a std
    derived from params, the gradient would be zero or meaningless, because the
    sampling operation itself is not differentiable w.r.t. its parameters.

    Holding eps fixed across different parameter values also gives us common
    random numbers in the search, which cuts the variance of the DR-vs-gradient
    comparison for free.

    Consequence: NO jax.random calls below this line. If you find yourself
    wanting one, the draw belongs in `draw_noise` instead.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .params import ImuParams


class BiasState(NamedTuple):
    """Evolving bias, carried across timesteps. SI units, same as the signal."""

    gyroscope: Array      # [rad / s],   shape (3,)
    accelerometer: Array  # [m / s^2],   shape (3,)


class NoiseDraw(NamedTuple):
    """Pre-drawn standard normals. Mean 0, variance 1, NO parameter dependence.

    Every field has shape (..., 3) for a single step, or (T, ..., 3) for a
    trace of T steps. These are the `eps` of the reparameterisation trick.
    """

    gyroscope_white: Array
    accelerometer_white: Array
    gyroscope_bias: Array
    accelerometer_bias: Array


class ImuSample(NamedTuple):
    """One IMU measurement."""

    gyroscope: Array      # [rad / s],  shape (3,)
    accelerometer: Array  # [m / s^2],  shape (3,)


def draw_noise(key: Array, shape: tuple[int, ...] = ()) -> NoiseDraw:
    """Draw the fixed standard normals for a rollout.

    This is the ONLY place jax.random appears in the noise model. Call it once
    per rollout, outside any gradient computation, and pass the result in.

    `shape` is the leading shape: () for a single step, (T,) for a trace of T
    steps, (B, T) for B parallel rollouts. The trailing 3 (axes) is added here.

    Note the key splitting: four independent noise sources need four
    independent keys. Reusing one key for all four would correlate the gyro
    white noise with the accel bias walk, which is physically wrong and is the
    kind of bug that produces a plausible-looking Allan curve.
    """
    k_gw, k_aw, k_gb, k_ab = jax.random.split(key, 4)
    full = (*shape, 3)
    return NoiseDraw(
        gyroscope_white=jax.random.normal(k_gw, full),
        accelerometer_white=jax.random.normal(k_aw, full),
        gyroscope_bias=jax.random.normal(k_gb, full),
        accelerometer_bias=jax.random.normal(k_ab, full),
    )


def initial_bias(params: ImuParams, key: Array | None = None) -> BiasState:
    """Bias at the start of an episode: the SEARCHED turn-on bias.

    Turn-on bias is a deterministic, signed, per-axis parameter rather than a
    random draw. That distinction is the point. A drawn bias would be nuisance
    randomness that the adversarial search has to average over; a searched
    parameter is something the search can target directly, and uniform DR can
    sample from the same box, so the two methods stay comparable.

    WHY THIS IS NOT ZERO ANY MORE. At an episode length of 20 s the turn-on
    bias bound is 60-190x the within-episode random-walk drift, making it the
    dominant bias term by a wide margin. Returning zero removed the single
    largest bias error source from the model, and with it the channel most
    likely to produce a real failure: a constant offset integrates into
    heading and attitude, whereas zero-mean per-step noise averages down as
    1/sqrt(N) over the episode.

    It also subsumes slow thermal drift. Over a seconds-long rollout a
    temperature-driven bias is indistinguishable from a constant offset, so
    one searched constant stands in for both and no separate thermal model is
    needed. (Temperature still matters elsewhere: a thermal ramp during a
    static Allan log fakes a random-walk slope at long tau, and sensitivity
    TCO argues for wider bounds than the room-temperature spec.)

    `key` is accepted and ignored, so switching to a drawn turn-on bias later
    would not change any call site.
    """
    del key  # see docstring
    return BiasState(
        gyroscope=jnp.broadcast_to(params.gyroscope_turn_on_bias, (3,)),
        accelerometer=jnp.broadcast_to(params.accelerometer_turn_on_bias, (3,)),
    )


def step(
    bias: BiasState,
    true_imu: ImuSample,
    params: ImuParams,
    eps: NoiseDraw,
    dt: float,
) -> tuple[BiasState, ImuSample]:
    """Corrupt one IMU sample and advance the bias by one timestep.

    This is the seam. It knows nothing about where `true_imu` came from -- a
    simulator, an analytic trajectory, or zeros for Allan validation.

    Discrete-time form of the Kalibr model. The two sqrt(dt) factors go in
    OPPOSITE directions, which is the single most important thing to check:

      White noise. The density sigma_nd is quoted per sqrt(Hz). Each sample
      averages the underlying process over dt, and a shorter window averages
      away less, so the per-sample standard deviation is sigma_nd / sqrt(dt).
      Sampling 4x faster raises per-sample noise 2x.

      Bias random walk. The variance of a random walk grows linearly in
      ELAPSED time, not in step count, so the per-step increment must shrink
      as dt shrinks: sigma_rw * sqrt(dt). After total time T the bias standard
      deviation is sigma_rw * sqrt(T) regardless of the dt chosen -- which is
      the invariant test_bias_random_walk_grows_as_sqrt_of_elapsed_time pins.

    Ordering: the measurement at step k uses the bias at step k, and the bias
    then advances. Updating first would introduce a one-sample lead that the
    Allan curve would not reveal.

    Scale factor multiplies (signal + bias) but NOT the additive white noise:
    a sensitivity error scales what the sensor is measuring, while the readout
    noise is added after that gain stage. This is the Kalibr convention.
    """
    # --- white noise: continuous density -> discrete per-sample std ---
    inv_sqrt_dt = 1.0 / jnp.sqrt(dt)
    gyro_white = params.gyroscope_noise_density * inv_sqrt_dt * eps.gyroscope_white
    accel_white = (
        params.accelerometer_noise_density * inv_sqrt_dt * eps.accelerometer_white
    )

    # --- measurement, using the bias at the START of this step ---
    measured = ImuSample(
        gyroscope=(1.0 + params.gyroscope_scale_factor)
        * (true_imu.gyroscope + bias.gyroscope)
        + gyro_white,
        accelerometer=(1.0 + params.accelerometer_scale_factor)
        * (true_imu.accelerometer + bias.accelerometer)
        + accel_white,
    )

    # --- bias diffusion: variance accumulates linearly in elapsed time ---
    sqrt_dt = jnp.sqrt(dt)
    next_bias = BiasState(
        gyroscope=bias.gyroscope
        + params.gyroscope_random_walk * sqrt_dt * eps.gyroscope_bias,
        accelerometer=bias.accelerometer
        + params.accelerometer_random_walk * sqrt_dt * eps.accelerometer_bias,
    )
    return next_bias, measured


def corrupt_trace(
    bias0: BiasState,
    true_imu: ImuSample,
    params: ImuParams,
    eps: NoiseDraw,
    dt: float,
) -> tuple[BiasState, ImuSample]:
    """Run `step` over a whole trace with jax.lax.scan.

    WHY SCAN AND NOT A PYTHON LOOP: a Python `for` over T steps would unroll
    into T copies of the graph at trace time. For the T=1e6 samples the Allan
    validation needs, that does not compile -- it exhausts memory first. `scan`
    compiles to a single loop primitive regardless of T, and its reverse-mode
    gradient is a single reversed loop rather than T stacked backward graphs.
    This is the difference between the Allan trace taking two seconds and not
    running at all.

    `true_imu` and `eps` are pytrees whose leaves have leading dimension T;
    scan slices all of them in lockstep, which is why they were defined as
    NamedTuples rather than loose arrays.

    To run B rollouts in parallel, vmap this -- do not add a batch dimension by
    hand. vmap over the batch axis composes with the scan over time correctly
    and keeps the single-rollout code readable.
    """

    def scan_fn(carry: BiasState, xs: tuple[ImuSample, NoiseDraw]):
        sample, eps_k = xs
        return step(carry, sample, params, eps_k, dt)

    return jax.lax.scan(scan_fn, bias0, (true_imu, eps))


def zero_trace(n: int) -> ImuSample:
    """A noise-free trace of a perfectly stationary device.

    Gyro reads zero. Accelerometer reads zero here TOO, not gravity: we want
    the Allan deviation of the noise process alone, and a constant gravity
    offset would add a DC term that the Allan variance is specifically designed
    to be blind to -- but only if it is exactly constant. Keeping it at zero
    removes the question entirely. Real bench data is a different matter; see
    fit_allan.py for how gravity is handled there.
    """
    z = jnp.zeros((n, 3))
    return ImuSample(gyroscope=z, accelerometer=z)
