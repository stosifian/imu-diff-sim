"""Modal app: run the GPU work from a shell instead of a notebook.

WHY THIS EXISTS. Colab is UI-only, so every dependency failure cost a
round-trip through a human pasting tracebacks. Five of those went on version
resolution alone. Modal is driven from the terminal, so the whole
edit-run-read-fix loop happens in one place. It also bills per second with no
idle charge, so a forgotten container costs nothing.

USAGE

    modal run modal_app.py                      # smoke: versions + GPU + env
    modal run modal_app.py::train --timesteps 2000000 --noise-level 0.0
    modal run modal_app.py::ofat  --policy /runs/go1_clean.pkl
    modal run modal_app.py::sh --cmd "python -m pytest -q"

The local repo is mounted, so edits here take effect on the next run with no
commit, push or pull.

PINS. Exactly the set verified working locally, and every one of them was
learned from a failed Colab run:

  jax 0.10.2   brax 0.14.2 (newest) calls jax.device_put_replicated, removed
               in JAX 0.11. No brax supports 0.11, so jax is what gets pinned.
  flax 0.12.8  0.12.10+ requires jax>=0.11.1, which would break the above;
               older than 0.12.8 calls jax.core.get_opaque_trace_state.
  one install  Separate pip calls re-resolve independently and silently undo
               each other's pins.
"""

import os

import modal

STACK = [
    "jax[cuda12]==0.10.2",
    "flax==0.12.8",
    "brax==0.14.2",
    "mujoco==3.16.0",
    "mujoco-mjx==3.16.0",
    "playground==0.2.0",
    "optax==0.2.8",
    "orbax-checkpoint==0.12.7",
    "numpy", "scipy", "matplotlib", "pyyaml", "scikit-learn", "pytest",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglew-dev", "libosmesa6-dev")
    # One pip_install call: the resolver must see every constraint at once.
    .pip_install(*STACK)
    .env({
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # mujoco_menagerie is fetched on first env creation (19 min on a cold
        # CPU). Point it at the volume so it is downloaded once, not per run.
        "MUJOCO_PLAYGROUND_CACHE": "/runs/.playground_cache",
    })
    .add_local_dir(".", remote_path="/root/imu-diff-sim", ignore=[
        ".git", ".venv", "__pycache__", "*.pyc", "reports/figures",
        "data", "runs", ".pytest_cache", "*.egg-info",
    ])
)

app = modal.App("imu-diff-sim", image=image)

# Survives between runs: checkpoints, sweep outputs, the menagerie cache.
vol = modal.Volume.from_name("imu-diff-sim-runs", create_if_missing=True)

GPU = os.environ.get("IMU_MODAL_GPU", "T4")
COMMON = dict(gpu=GPU, volumes={"/runs": vol}, timeout=60 * 60 * 4)


#: Lines MJX/Warp emit per physics step. At training rates this floods stdout
#: hard enough that Modal drops output ("exceeded output rate limit"), which
#: swallowed the progress lines and the throughput number on the first run.
#: The solver-iteration warning is benign -- the Playground Go1 config uses a
#: low iteration count deliberately -- so it is dropped rather than fixed.
_NOISE = (
    "solver iterations limit reached",
    "linesearch iterations limit reached",
    "To disable the print warning",
    "load on device",
    "Module ",
)


def _run(cmd: str) -> int:
    """Run a command in the repo, filtering known log spam, and commit /runs."""
    import subprocess
    import sys
    import time

    pattern = "|".join(n.replace(" ", "[[:space:]]") for n in _NOISE)
    wrapped = f"set -o pipefail; {cmd} 2>&1 | grep -a -v -E '{pattern}'"

    print(f"$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(
        wrapped, shell=True, cwd="/root/imu-diff-sim", executable="/bin/bash"
    ).returncode
    elapsed = time.time() - t0
    vol.commit()  # persist /runs before the container dies
    print(f"\n[{elapsed:.1f}s wall, exit {rc}]", flush=True)
    if rc != 0:
        print(f"FAILED: exit {rc}", file=sys.stderr, flush=True)
    return rc


@app.function(**COMMON)
def sh(cmd: str) -> int:
    """Escape hatch: run any command on the GPU box. The debugging workhorse."""
    return _run(cmd)


@app.function(**COMMON)
def smoke() -> int:
    """Versions, GPU, and a Go1 env creation. Everything Colab got wrong."""
    import jax, jaxlib, flax, brax

    for n, v in (("jax", jax.__version__), ("jaxlib", jaxlib.__version__),
                 ("flax", flax.__version__), ("brax", brax.__version__)):
        print(f"{n:8s} {v}")
    devs = jax.devices()
    print("devices:", devs)
    assert any(d.platform == "gpu" for d in devs), "no GPU visible to JAX"

    import time
    from mujoco_playground import registry
    t0 = time.time()
    env = registry.load("Go1JoystickFlatTerrain")
    print(f"Go1 env created in {time.time()-t0:.1f}s")
    print("obs sizes:", dict(env.observation_size))
    print("ctrl dt  :", env.dt)
    vol.commit()
    return 0


@app.function(**COMMON)
def train(timesteps: int = 200_000_000, noise_level: float = 0.0,
          name: str = "go1_clean", env: str = "Go1JoystickFlatTerrain") -> int:
    return _run(
        f"python scripts/train_go1.py --env {env} --timesteps {timesteps} "
        f"--noise-level {noise_level} --out /runs/{name}.pkl"
    )


@app.function(**COMMON)
def ofat(policy: str = "/runs/go1_clean.pkl", episodes: int = 64,
         horizon: int = 1000, points: int = 9,
         env: str = "Go1JoystickFlatTerrain") -> int:
    return _run(
        f"python scripts/ofat_sensitivity.py --env {env} --policy {policy} "
        f"--episodes {episodes} --horizon {horizon} --points {points} "
        f"--out /runs/ofat_{policy.split('/')[-1].replace('.pkl','')}.npz"
    )


@app.function(**COMMON)
def tests() -> int:
    return _run("python -m pytest -q --no-header")


@app.local_entrypoint()
def main():
    """Default: the smoke test. Cheap, and it fails where Colab failed."""
    raise SystemExit(smoke.remote())
