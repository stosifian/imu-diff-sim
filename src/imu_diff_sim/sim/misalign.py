"""IMU-to-body misalignment: the parameter space the measurements point at.

WHY THIS AND NOT NOISE DENSITY. The OFAT sweep found every gyro noise channel
clear by 25-635x, while attitude error degrades the policy from ~4.5 deg and
collapses it by ~11 deg. A realistic in-field stack-up -- datasheet sources
RSS 1.41 deg plus a 3 deg mechanical mounting error -- lands at 4.41 deg,
essentially on the degradation onset. So a failure boundary genuinely exists
inside a defensible physical bound, which is what the original search
experiment needed and never had over noise parameters.

PHYSICAL MODEL. A mounting error rotates the whole sensor package, so the
same rotation applies to BOTH the gyro and the accelerometer-derived gravity
vector -- they are one chip on one board. That coupling is what makes this
three parameters rather than six, and it is why injecting the two channels
independently (as the OFAT factors did) is not the physical error mode.

Parameterised as a rotation vector r in radians: direction is the axis,
magnitude is the angle. Applied with the full Rodrigues formula rather than
the small-angle approximation, so the model stays exact out to the box edge
and stays differentiable everywhere (the 1/theta terms are guarded).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

#: Per-axis bound in radians. 5 deg is the datasheet stack-up (1.41 deg RSS
#: from zero-g offset, TCO over 40 K, cross-axis and differential TCS) plus a
#: ~3.5 deg mechanical mounting tolerance. Every point inside is a device
#: someone could ship; the box corner (three axes at 5 deg, RSS 8.7 deg)
#: reaches into the measured failure region, so the boundary is enclosed.
MISALIGN_BOUND_RAD: float = 5.0 * jnp.pi / 180.0


def rodrigues(v: Array, r: Array) -> Array:
    """Rotate v by the rotation vector r (axis * angle, radians).

    Written WITHOUT dividing by theta. The textbook form needs k = r/theta,
    which is singular at r = 0; an earlier version guarded that with
    jnp.where(theta2 < eps, v, rotated), and that guard was a bug: where
    routes the gradient through the selected branch only, so d/dr was exactly
    ZERO at r = 0 -- which is precisely where every search restart begins.
    The spike measured ||g|| = 0 at every horizon because of it.

    Rewriting in terms of r directly removes the singularity instead of
    hiding it:

        v' = v cos(th) + (r x v) sinc(th) + r (r.v) (1 - cos th)/th^2

    with sin(th)/th and (1 - cos th)/th^2 evaluated through jnp.sinc, which
    is smooth and differentiable at zero. The identity
    (1 - cos th)/th^2 = 0.5 sinc(th/2)^2 avoids the second division.
    """
    theta2 = jnp.sum(r * r)
    theta = jnp.sqrt(theta2 + 1e-30)
    # jnp.sinc(x) is sin(pi x)/(pi x), so sinc(th/pi) = sin(th)/th.
    sinc_t = jnp.sinc(theta / jnp.pi)
    half_sinc = jnp.sinc(theta / (2.0 * jnp.pi))
    return (
        v * jnp.cos(theta)
        + jnp.cross(r, v) * sinc_t
        + r * jnp.dot(r, v) * 0.5 * half_sinc ** 2
    )


def apply(gyro: Array, gravity: Array, r: Array) -> tuple[Array, Array]:
    """Apply one mounting misalignment to both IMU-derived channels."""
    return rodrigues(gyro, r), rodrigues(gravity, r)


def to_unconstrained(r: Array) -> Array:
    """Physical rotation vector -> unbounded search coordinates."""
    return jnp.arctanh(jnp.clip(r / MISALIGN_BOUND_RAD, -0.999999, 0.999999))


def from_unconstrained(u: Array) -> Array:
    """Unbounded search coordinates -> physical rotation vector.

    tanh-bounded per axis, same structural-constraint approach as params.py:
    any real vector maps to a physically plausible mounting error, so no
    search method ever needs a clip.
    """
    return MISALIGN_BOUND_RAD * jnp.tanh(u)
