"""OFAT sensitivity sweep: how much IMU error does the Go1 policy tolerate?

WHY THIS RUNS BEFORE ANY SENSOR MODELLING. The project's premise is that
failure modes exist inside a physically plausible IMU parameter space. That
premise is testable directly, without a sensor model, by injecting idealised
error modes one at a time and finding where performance knees. Then compare
each knee against what a real BMI270 produces.

If no channel's knee is within reach of the real device, the premise is false
for this task and policy -- and that is worth learning in an afternoon rather
than in week three. If some channel IS reachable, the sweep says which one,
and the sensor model gets built where it matters.

The built-in observation DR is DISABLED (noise_config.level = 0). Measuring an
injected 1e-3 rad/s bias on top of the stock +-0.2 rad/s uniform noise would
measure nothing.

FACTORS, chosen to separate magnitude from structure:
  gyro_white    zero-mean i.i.d., redrawn each step. The channel the env's DR
                already covers, and the one a policy can average away.
  gyro_bias     constant for the episode. Same units, but cannot be averaged
                away and integrates into heading. Absent from the env's DR.
  gyro_scale    multiplicative. Structurally absent from additive DR.
  gravity_tilt  constant tilt of the gravity unit vector -- the accelerometer
                equivalent of a persistent bias.
  latency       whole-step delay on gyro and gravity. Often the dominant
                real-hardware IMU-path error and absent from the env's DR.

Comparing gyro_white against gyro_bias at equal magnitude is the direct test
of the structural claim: if bias degrades performance at far lower magnitude,
the 1/sqrt(N) averaging argument is confirmed empirically.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

FACTORS = ("gyro_white", "gyro_bias", "gyro_scale", "gravity_tilt", "latency")


def build_rollout(env, inference_fn, cfg):
    """Return rollout(key, factor, magnitude) -> metrics dict."""
    import jax
    import jax.numpy as jnp

    from imu_diff_sim.sim.go1_obs import GRAVITY, GYRO

    horizon = cfg["horizon"]
    max_lat = cfg["max_latency"]

    def rollout(key, factor_id, magnitude):
        k_env, k_bias, k_noise, k_act = jax.random.split(key, 4)
        state = env.reset(k_env)

        # Per-episode constants. Drawn once, held for the whole rollout --
        # that persistence is the entire point of the bias/scale factors.
        bias_dir = jax.random.normal(k_bias, (3,))
        bias_dir = bias_dir / (jnp.linalg.norm(bias_dir) + 1e-9)
        white = jax.random.normal(k_noise, (horizon, 3))

        obs0 = state.obs["state"] if isinstance(state.obs, dict) else state.obs
        lat_buf = jnp.tile(obs0[GYRO], (max_lat, 1))
        grav_buf = jnp.tile(obs0[GRAVITY], (max_lat, 1))

        def corrupt(obs, t, lat_buf, grav_buf):
            # jnp.where rather than Python branching: factor_id is traced.
            gyro = obs[GYRO]
            gyro = jnp.where(factor_id == 0, gyro + magnitude * white[t], gyro)
            gyro = jnp.where(factor_id == 1, gyro + magnitude * bias_dir, gyro)
            gyro = jnp.where(factor_id == 2, gyro * (1.0 + magnitude), gyro)
            grav = obs[GRAVITY]
            tilted = grav + magnitude * bias_dir
            tilted = tilted / (jnp.linalg.norm(tilted) + 1e-9)
            grav = jnp.where(factor_id == 3, tilted, grav)

            # Latency: integer delay in whole control steps. Applied to both
            # IMU-derived channels, since on hardware they share a path.
            lat_buf = jnp.concatenate([lat_buf[1:], gyro[None, :]])
            grav_buf = jnp.concatenate([grav_buf[1:], grav[None, :]])
            d = jnp.clip(magnitude.astype(jnp.int32), 0, max_lat - 1)
            delayed_g = lat_buf[max_lat - 1 - d]
            delayed_v = grav_buf[max_lat - 1 - d]
            gyro = jnp.where(factor_id == 4, delayed_g, gyro)
            grav = jnp.where(factor_id == 4, delayed_v, grav)

            return obs.at[GYRO].set(gyro).at[GRAVITY].set(grav), lat_buf, grav_buf

        def step(carry, t):
            state, lat_buf, grav_buf, key = carry
            key, sub = jax.random.split(key)
            raw = state.obs["state"] if isinstance(state.obs, dict) else state.obs
            obs, lat_buf, grav_buf = corrupt(raw, t, lat_buf, grav_buf)
            inp = dict(state.obs) if isinstance(state.obs, dict) else obs
            if isinstance(state.obs, dict):
                inp["state"] = obs
            act, _ = inference_fn(inp, sub)
            state = env.step(state, act)

            cmd = raw[45:48]
            linvel = raw[0:3]
            gyro_true = raw[3:6]
            track = jnp.sum((linvel[:2] - cmd[:2]) ** 2) + (gyro_true[2] - cmd[2]) ** 2
            return (state, lat_buf, grav_buf, key), (
                state.reward, state.done, track
            )

        (_, _, _, _), (rew, done, track) = jax.lax.scan(
            step, (state, lat_buf, grav_buf, k_act), jnp.arange(horizon)
        )
        # Everything after the first termination is not meaningful; mask it.
        alive = jnp.cumprod(1.0 - done)
        n_alive = jnp.sum(alive)
        return {
            "return": jnp.sum(rew * alive),
            "tracking_mse": jnp.sum(track * alive) / jnp.maximum(n_alive, 1.0),
            "fell": jnp.max(done),
            "survived": n_alive / horizon,
        }

    return rollout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--factors", nargs="+", default=list(FACTORS), choices=FACTORS)
    ap.add_argument("--out", type=Path, default=Path("runs/ofat.npz"))
    args = ap.parse_args()

    import functools

    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry, wrapper

    from imu_diff_sim.sim.go1_obs import (
        BMI270_GYRO_WHITE_STD_50HZ,
        PLAYGROUND_GRAVITY_DR_HALFWIDTH,
        PLAYGROUND_GYRO_DR_STD,
    )

    print(f"JAX devices: {jax.devices()}")
    blob = pickle.loads(args.policy.read_bytes())

    cfg = registry.get_default_config(args.env)
    cfg.noise_config.level = 0.0  # critical: measure against a clean floor
    env = registry.load(args.env, config=cfg)
    env = wrapper.wrap_for_brax_training(
        env, episode_length=args.horizon, action_repeat=1
    )

    ppo_params = blob.get("ppo_params", {})
    net_kwargs = dict(ppo_params.get("network_factory", {}) or {})
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        **net_kwargs,
    )
    inference_fn = ppo_networks.make_inference_fn(networks)(
        blob["params"], deterministic=True
    )

    rollout = build_rollout(env, inference_fn,
                            {"horizon": args.horizon, "max_latency": 8})
    batched = jax.jit(jax.vmap(rollout, in_axes=(0, None, None)))

    # Sweep ranges. Gyro ranges deliberately span from well below BMI270
    # (7.9e-4 rad/s) to above the env's own DR (0.115 rad/s), so both
    # reference lines fall inside the plot.
    sweeps = {
        "gyro_white": np.logspace(-4.5, -0.3, args.points),
        "gyro_bias": np.logspace(-4.5, -0.3, args.points),
        "gyro_scale": np.logspace(-3, np.log10(0.5), args.points),
        "gravity_tilt": np.logspace(-3.5, -0.3, args.points),
        "latency": np.arange(0, min(args.points, 8), dtype=float),
    }

    keys = jax.random.split(jax.random.PRNGKey(0), args.episodes)
    results = {}
    for factor in args.factors:
        fid = FACTORS.index(factor)
        mags, rows = sweeps[factor], []
        print(f"\n{factor}")
        print(f"  {'magnitude':>12} {'return':>10} {'track mse':>11} "
              f"{'fall rate':>10} {'survived':>9}")
        for m in mags:
            out = batched(keys, fid, jnp.asarray(m))
            row = {k: float(jnp.mean(v)) for k, v in out.items()}
            rows.append(row)
            print(f"  {m:>12.4g} {row['return']:>10.2f} {row['tracking_mse']:>11.4f} "
                  f"{row['fell']:>10.2f} {row['survived']:>9.2f}")
        results[factor] = {"magnitudes": mags,
                           **{k: np.array([r[k] for r in rows]) for k in rows[0]}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{f"{f}__{k}": v for f, d in results.items()
                          for k, v in d.items()})
    print(f"\nsaved -> {args.out}")

    print("\nREFERENCE MAGNITUDES")
    print(f"  BMI270 gyro white noise @50Hz : {BMI270_GYRO_WHITE_STD_50HZ:.3g} rad/s")
    print(f"  Playground gyro DR (std)      : {PLAYGROUND_GYRO_DR_STD:.3g} rad/s")
    print(f"  Playground gravity DR         : {PLAYGROUND_GRAVITY_DR_HALFWIDTH:.3g}")
    print("\nThe decision is the gap between each factor's knee and the BMI270 line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
