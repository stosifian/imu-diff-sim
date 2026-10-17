"""Train a Go1 joystick policy with MuJoCo Playground + brax PPO.

GPU ONLY. This will not finish on CPU in any useful time -- creating the env
alone took 19 minutes on an M1. The script asserts a GPU is present rather
than silently running for days.

Two noise regimes, selected with --noise-level:

  1.0  (default) the stock config: uniform gyro DR of +-0.2 rad/s, which is
       146x a real BMI270 at 50 Hz, and identical across seven Playground
       robots. This is the "standard practice" baseline.
  0.0  no observation noise at all. Use this for the OFAT sweep, so injected
       errors are measured against a clean floor instead of on top of a noise
       source 146x larger than anything being injected.

Checkpoints are written every save interval. On Colab point --out at a
Google Drive path: a disconnect wipes the VM and a three-hour run with it.
"""

from __future__ import annotations

import argparse
import functools
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--timesteps", type=int, default=60_000_000)
    ap.add_argument("--noise-level", type=float, default=1.0,
                    help="0.0 disables the env's built-in observation DR")
    ap.add_argument("--out", type=Path, default=Path("runs/go1_policy.pkl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    import jax

    devs = jax.devices()
    kinds = {d.platform for d in devs}
    print(f"JAX devices: {devs}")
    if "gpu" not in kinds and "tpu" not in kinds and not args.allow_cpu:
        print(
            "\nNO GPU DETECTED. Training on CPU is not viable here.\n"
            "On Colab: Runtime > Change runtime type > GPU, then\n"
            "  pip install -U 'jax[cuda12]'\n"
            "and restart the runtime. Pass --allow-cpu to override.",
            file=sys.stderr,
        )
        return 2

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    env_cfg = registry.get_default_config(args.env)
    env_cfg.noise_config.level = args.noise_level
    print(f"observation noise level: {env_cfg.noise_config.level}")
    print(f"  gyro DR    : +-{env_cfg.noise_config.scales.gyro * env_cfg.noise_config.level}")
    print(f"  gravity DR : +-{env_cfg.noise_config.scales.gravity * env_cfg.noise_config.level}")

    env = registry.load(args.env, config=env_cfg)
    eval_env = registry.load(args.env, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(args.env)
    ppo_params.num_timesteps = args.timesteps

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **dict(ppo_params.network_factory or {}),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"  {step:>12,} steps  reward {r:8.2f}  [{time.time()-t0:6.0f}s]", flush=True)

    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in ppo_params.items()
           if k not in ("network_factory", "num_timesteps")},
        num_timesteps=ppo_params.num_timesteps,
        network_factory=network_factory,
        progress_fn=progress,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

    with args.out.open("wb") as f:
        pickle.dump(
            {"params": params, "env": args.env, "noise_level": args.noise_level,
             "ppo_params": ppo_params.to_dict()}, f)
    print(f"\nsaved -> {args.out}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
