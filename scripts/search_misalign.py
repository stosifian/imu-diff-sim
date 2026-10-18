"""Two-arm search over IMU mounting misalignment: uniform DR vs CEM.

WHAT THIS TESTS, AND WHAT IT DOES NOT. The brief's claim was that
gradient-informed search finds a policy's failure modes more efficiently than
uniform sampling. That claim cannot be tested here: AD through MJX contact is
correct only to H=10 (0.2 s), far shorter than a tilt-induced fall takes to
develop, and randomized smoothing over 32 episodes leaves the mean gradient
statistically indistinguishable from zero (scripts/spike_mjx_grad.py). So the
gradient arm does not exist on this system.

What remains is the honest weaker question: does an ADAPTIVE gradient-free
method find the failure boundary more efficiently than uniform sampling? Same
budget, same box, same failure definition.

WHY THIS PARAMETER SPACE. The OFAT sweep put every gyro noise channel 25-635x
clear of anything that degrades the policy, while attitude error degrades it
from ~4.5 deg. A mounting error rotates the whole sensor package, so one
rotation vector drives both the gyro and the gravity channel -- three
parameters, bounded at 5 deg/axis from the datasheet stack-up plus a stated
mechanical tolerance. The failure boundary is inside the box; the measured
degradation onset sits at a realistic in-field stack-up of 4.41 deg.

BUDGET is counted in simulator steps, not in samples, and both arms are
charged identically. Common random numbers: both arms evaluate against the
same episode seeds, so the comparison is not confounded by sampling luck.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

#: A configuration counts as a failure if it costs this fraction of the
#: unperturbed return. Fixed before running; reported alongside results.
FAIL_FRACTION = 0.90


def build_eval(env, infer, horizon: int):
    import jax
    import jax.numpy as jnp

    from imu_diff_sim.sim import misalign as M
    from imu_diff_sim.sim.go1_obs import GRAVITY, GYRO

    def rollout(key, r):
        def step(carry, _):
            state, k = carry
            k, sub = jax.random.split(k)
            obs = dict(state.obs)
            raw = obs["state"]
            gyro, grav = M.apply(raw[GYRO], raw[GRAVITY], r)
            obs["state"] = raw.at[GYRO].set(gyro).at[GRAVITY].set(grav)
            act, _ = infer(obs, sub)
            state = env.step(state, act)
            return (state, k), (state.reward, state.done)

        state0 = env.reset(key)
        _, (rew, done) = jax.lax.scan(step, (state0, key), None, length=horizon)
        alive = jnp.cumprod(1.0 - done) > 0
        return jnp.sum(jnp.where(alive, jnp.nan_to_num(rew), 0.0))

    return jax.jit(jax.vmap(rollout, in_axes=(0, None)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--budget", type=int, default=240,
                    help="parameter evaluations per arm (equal for both)")
    ap.add_argument("--episodes", type=int, default=16,
                    help="episodes averaged per evaluation")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--cem-pop", type=int, default=24)
    ap.add_argument("--cem-elite", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/runs/search_misalign.npz"))
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    from imu_diff_sim.compat import patch_device_put_replicated
    from imu_diff_sim.sim import misalign as M

    patch_device_put_replicated()
    print(f"devices: {jax.devices()}")

    blob = pickle.loads(args.policy.read_bytes())
    cfg = registry.get_default_config(args.env)
    cfg.noise_config.level = 0.0
    env = registry.load(args.env, config=cfg)
    pp = blob.get("ppo_params", {})
    normalize = (running_statistics.normalize
                 if pp.get("normalize_observations", True)
                 else types.identity_observation_preprocessor)
    nets = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=env.action_size,
        preprocess_observations_fn=normalize,
        **dict(pp.get("network_factory", {}) or {}))
    infer = ppo_networks.make_inference_fn(nets)(blob["params"], deterministic=True)

    batched = build_eval(env, infer, args.horizon)
    # Common random numbers: both arms see the same episodes.
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)

    def evaluate(r_phys) -> float:
        return float(jnp.mean(batched(keys, jnp.asarray(r_phys))))

    base = evaluate(np.zeros(3))
    thresh = FAIL_FRACTION * base
    print(f"baseline return {base:.2f} over {args.episodes} episodes x "
          f"{args.horizon} steps")
    print(f"failure threshold: return < {thresh:.2f} "
          f"({FAIL_FRACTION:.0%} of baseline)")
    print(f"box: +-{np.degrees(M.MISALIGN_BOUND_RAD):.1f} deg per axis, "
          f"budget {args.budget} evaluations per arm\n")

    rng = np.random.default_rng(args.seed)
    B = float(M.MISALIGN_BOUND_RAD)
    results = {}

    # --- Arm A: uniform domain randomization -------------------------------
    t0 = time.time()
    pts, vals = [], []
    for _ in range(args.budget):
        r = rng.uniform(-B, B, 3)
        pts.append(r); vals.append(evaluate(r))
    results["uniform"] = (np.array(pts), np.array(vals))
    nf = int(np.sum(np.array(vals) < thresh))
    print(f"uniform DR : {nf:>4}/{args.budget} failures  "
          f"worst {np.min(vals):.2f}  [{time.time()-t0:.0f}s]")

    # --- Arm B: cross-entropy method ---------------------------------------
    # Adaptive but gradient-free: fit a Gaussian to the worst-performing
    # elites each generation and resample. The arm the brief's gradient
    # method would have had to beat.
    t0 = time.time()
    mu, sigma = np.zeros(3), np.full(3, B / 2.0)
    pts, vals = [], []
    gens = args.budget // args.cem_pop
    for g in range(gens):
        pop = np.clip(rng.normal(mu, sigma, (args.cem_pop, 3)), -B, B)
        v = np.array([evaluate(p) for p in pop])
        pts.extend(pop); vals.extend(v)
        elite = pop[np.argsort(v)[: args.cem_elite]]   # lowest return = worst
        mu, sigma = elite.mean(0), elite.std(0) + 1e-4
        print(f"  CEM gen {g+1}/{gens}: best-found return {np.min(vals):.2f}, "
              f"|mu| {np.degrees(np.linalg.norm(mu)):.2f} deg, "
              f"sigma {np.degrees(sigma.mean()):.2f} deg")
    results["cem"] = (np.array(pts), np.array(vals))
    nf = int(np.sum(np.array(vals) < thresh))
    print(f"CEM        : {nf:>4}/{len(vals)} failures  "
          f"worst {np.min(vals):.2f}  [{time.time()-t0:.0f}s]")

    out = {"baseline": base, "threshold": thresh, "bound_rad": B,
           "budget": args.budget, "episodes": args.episodes,
           "horizon": args.horizon, "fail_fraction": FAIL_FRACTION}
    for k, (p_, v_) in results.items():
        out[f"{k}__points"] = p_
        out[f"{k}__returns"] = v_
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
