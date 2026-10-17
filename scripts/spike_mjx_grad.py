"""Can jax.grad flow through an MJX Go1 rollout at all?

The week-1 spike validated gradients on Brax positional, not MJX, and the
Modal container runs MuJoCo's Warp backend rather than classic MJX. Whether
reverse-mode AD works through that path is unverified, and the entire
gradient arm of the three-arm comparison depends on it.

Checked before building anything on top:
  C1 the gradient exists and is finite
  C2 it agrees with central finite differences
  C3 its sign is stable across independent episodes
  C4 ||g|| does not grow faster than ~linearly in horizon

If AD does not work here the comparison becomes DR vs CEM, which is still
worth running but is not the brief's claim -- and that is a finding to report
rather than a setback to hide.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--horizons", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--smooth", type=int, default=0,
                    help="episodes to average for the randomized-smoothing test")
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"],
                    help="MJX implementation. Go1 defaults to warp, whose JAX "
                         "FFI call has no transpose rule, so reverse-mode AD "
                         "fails outright. 'jax' is pure JAX and differentiable "
                         "in principle -- slower, but the gradient arm needs it.")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    from imu_diff_sim.compat import patch_device_put_replicated
    from imu_diff_sim.sim import misalign as M
    from imu_diff_sim.sim.go1_obs import GRAVITY, GYRO

    patch_device_put_replicated()
    print(f"devices: {jax.devices()}")

    blob = pickle.loads(args.policy.read_bytes())
    cfg = registry.get_default_config(args.env)
    cfg.noise_config.level = 0.0
    cfg.impl = args.impl
    print(f"MJX impl: {cfg.impl}")
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

    def loss(u, key, horizon):
        """Smooth surrogate: mean squared velocity-tracking error.

        Not the env reward -- that is clipped and contains discrete terms.
        Not fall rate -- a step function has no gradient.
        """
        r = M.from_unconstrained(u)

        def step(carry, _):
            state, k = carry
            k, sub = jax.random.split(k)
            obs = dict(state.obs)
            raw = obs["state"]
            gyro, grav = M.apply(raw[GYRO], raw[GRAVITY], r)
            obs["state"] = raw.at[GYRO].set(gyro).at[GRAVITY].set(grav)
            act, _ = infer(obs, sub)
            state = env.step(state, act)
            cmd, lin = raw[45:48], raw[0:3]
            err = jnp.sum((lin[:2] - cmd[:2]) ** 2) + (raw[5] - cmd[2]) ** 2
            return (state, k), err

        state0 = env.reset(key)
        _, errs = jax.lax.scan(step, (state0, key), None, length=horizon)
        return jnp.mean(errs)

    u0 = jnp.zeros(3)
    print(f"\n{'H':>5}{'C1 finite':>11}{'C2 rel err':>12}{'C3 sign':>9}"
          f"{'||g||':>12}{'secs':>7}")
    print("-" * 56)
    norms, hs = [], []
    for H in args.horizons:
        t0 = time.time()
        try:
            gfn = jax.jit(jax.grad(loss), static_argnums=(2,))
            lfn = jax.jit(loss, static_argnums=(2,))
            G = np.stack([np.asarray(gfn(u0, jax.random.PRNGKey(i), H))
                          for i in range(args.draws)])
        except Exception as exc:  # noqa: BLE001
            print(f"{H:>5}  AD FAILED: {type(exc).__name__}: {str(exc)[:70]}")
            continue

        c1 = bool(np.all(np.isfinite(G)))
        h = 1e-3
        rel = []
        for i in range(3):
            e = jnp.zeros(3).at[i].set(h)
            num = float((lfn(u0 + e, jax.random.PRNGKey(0), H)
                         - lfn(u0 - e, jax.random.PRNGKey(0), H)) / (2 * h))
            ana = float(G[0, i])
            rel.append(abs(ana - num) / max(abs(num), abs(ana), 1e-30))
        med = float(np.median(rel))
        sg = np.sign(G)
        maj = np.sign(np.sum(sg, axis=0)); maj[maj == 0] = 1
        agree = float(np.mean(sg == maj[None, :]))
        nrm = float(np.median(np.linalg.norm(G, axis=1)))
        norms.append(nrm); hs.append(H)
        print(f"{H:>5}{'ok' if c1 else 'FAIL':>11}{med:>11.1%}"
              f"{agree:>9.2f}{nrm:>12.3e}{time.time()-t0:>7.0f}")

    # Randomized smoothing: does averaging over episodes recover a usable
    # direction? Individual gradients being noisy is survivable if their mean
    # is statistically distinguishable from zero; if it is not, there is no
    # descent direction to find and the gradient arm cannot exist.
    if args.smooth:
        print(f"\n{'H':>5}{'n':>5}{'|mean|/SEM per axis':>34}{'verdict':>12}")
        print("-" * 56)
        for H in args.horizons:
            gfn = jax.jit(jax.grad(loss), static_argnums=(2,))
            G = np.stack([np.asarray(gfn(u0, jax.random.PRNGKey(1000 + i), H))
                          for i in range(args.smooth)])
            mean, sem = G.mean(0), G.std(0) / np.sqrt(args.smooth)
            z = np.abs(mean) / np.maximum(sem, 1e-30)
            ok = bool(np.max(z) > 3.0)
            print(f"{H:>5}{args.smooth:>5}"
                  f"{'  '.join(f'{v:6.2f}' for v in z):>34}"
                  f"{'usable' if ok else 'no signal':>12}")

    if len(norms) >= 2:
        k = float(np.polyfit(np.log(hs), np.log(np.maximum(norms, 1e-30)), 1)[0])
        print(f"\nC4: ||g|| ~ H^{k:.2f}  ->  {'ok' if k <= 1.5 else 'FAIL (chaotic)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
