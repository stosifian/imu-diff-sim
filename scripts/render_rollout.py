"""Render the frozen Go1 policy walking, for the write-up's opening figure.

WHY THIS EXISTS. Every number in the write-up is a measurement on one policy
doing one task, and the reader meets it as "the policy" in section 4 with no
picture of what it is. Ten seconds of video costs nothing and makes the
48-dimensional observation vector concrete.

This is illustration, not measurement. The command is pinned rather than
resampled on the env's own schedule, so the clip shows steady locomotion
instead of the stop-start the training distribution produces. The built-in
observation DR is disabled, matching every other run in the project.

The policy construction MUST mirror ofat_sensitivity.py: brax PPO trains this
task with normalize_observations=True, so the saved params are
(normalizer_stats, policy_weights) and building the network with the default
identity preprocessor silently produces a policy that falls over. The
--min-survival gate below catches that, the same way the OFAT baseline check
does.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--policy", type=Path, default=Path("runs/go1_clean.pkl"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures/go1-flat.mp4"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500, help="50 Hz, so 500 = 10 s")
    ap.add_argument("--render-every", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera", default="track")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="scale the camera offset; <1 moves it closer")
    ap.add_argument("--light", type=float, nargs=2, default=None,
                    metavar=("AMBIENT", "DIFFUSE"),
                    help="headlight override. The rough-terrain scene ships "
                         "at half the flat scene's (0.1/0.4 against 0.2/0.8), "
                         "which renders too dark to read beside it.")
    ap.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0),
                    metavar=("VX", "VY", "WZ"),
                    help="pinned joystick command [m/s, m/s, rad/s]")
    ap.add_argument("--noise-level", type=float, default=0.0)
    ap.add_argument("--min-survival", type=float, default=1.0,
                    help="fraction of --steps the policy must stay up for")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from imu_diff_sim.compat import patch_device_put_replicated

    patch_device_put_replicated()

    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    print(f"JAX devices: {jax.devices()}")
    blob = pickle.loads(args.policy.read_bytes())

    cfg = registry.get_default_config(args.env)
    cfg.noise_config.level = args.noise_level
    env = registry.load(args.env, config=cfg)

    if args.light is not None:
        env.mj_model.vis.headlight.ambient[:] = args.light[0]
        env.mj_model.vis.headlight.diffuse[:] = args.light[1]

    if args.zoom != 1.0:
        # The tracking cameras are trackcom with a fixed body-frame offset, so
        # scaling that offset is the only zoom control the scene exposes.
        import mujoco

        cam = mujoco.mj_name2id(
            env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera
        )
        env.mj_model.cam_pos[cam] *= args.zoom

    ppo_params = blob.get("ppo_params", {})
    net_kwargs = dict(ppo_params.get("network_factory", {}) or {})
    normalize = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", True)
        else types.identity_observation_preprocessor
    )
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=normalize,
        **net_kwargs,
    )
    inference_fn = ppo_networks.make_inference_fn(networks)(
        blob["params"], deterministic=True
    )

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(inference_fn)

    cmd = jnp.asarray(args.command)
    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    state = reset(k)
    state.info["command"] = cmd

    traj, reward, survived = [state], 0.0, args.steps
    for i in range(args.steps):
        rng, k = jax.random.split(rng)
        action, _ = act(state.obs, k)
        state = step(state, action)
        # Pin the command: the env resamples it on its own schedule, which
        # makes the clip stop and turn for reasons the viewer cannot see.
        state.info["command"] = cmd
        reward += float(state.reward)
        traj.append(state)
        if float(state.done) > 0.5:
            survived = i + 1
            print(f"terminated at step {survived} ({survived * env.dt:.2f} s)")
            break

    frac = survived / args.steps
    print(f"survived {survived}/{args.steps} steps, return {reward:.2f}")
    if frac < args.min_survival:
        print(
            f"\nFAILED: policy stayed up for only {frac:.0%} of the clip.\n"
            "  1. preprocess_observations_fn must match training "
            "(normalize_observations)\n"
            "  2. the command may be outside the training distribution\n"
            "  3. wrong checkpoint for this env\n"
            "Lower --min-survival to render anyway.",
            file=sys.stderr,
        )
        return 1

    frames = env.render(
        traj[:: args.render_every], height=args.height, width=args.width,
        camera=args.camera,
    )
    fps = 1.0 / (env.dt * args.render_every)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import mediapy

    mediapy.write_video(args.out, np.asarray(frames), fps=fps)
    print(f"{len(frames)} frames at {fps:.0f} fps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
