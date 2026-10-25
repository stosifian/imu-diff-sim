"""Gradient spike: are AD gradients through the sim usable at all?

THE QUESTION. The search arms assume d(task_loss)/d(IMU_params) computed by
jax.grad through a contact-rich rollout is informative enough to optimize
against. That assumption is not safe. Reverse-mode AD through contact is known
to give high-variance or effectively meaningless gradients (Suh et al., ICML
2022, "Do Differentiable Simulators Give Better Policy Gradients?"; Metz et al.
on chaotic unrolls). A gradient can be finite, agree with finite differences at
one point, and still be useless for optimization because the landscape is
chaotic at that horizon.

PASS/FAIL CRITERION, FIXED BEFORE THE FIRST RUN. For each (backend, horizon):

  C1 FINITE        every component of the AD gradient is finite.
  C2 ACCURACY      median relative error against central finite differences
                   is below 20%.
  C3 SIGN STABILITY across >= 8 independent noise draws, the fraction of draws
                   agreeing with the majority sign exceeds 0.8.
  C4 CONDITIONING  ||g|| does not grow faster than ~linearly in H. Exponential
                   growth is the signature of a chaotic unroll.

A backend/horizon passing all four is usable. The LONGEST horizon that passes
is what the search should adopt. If nothing passes, that is a real result: report
the horizon at which analytic gradients stop being informative and run the
three-arm comparison there.

CAVEAT, STATED UP FRONT. The policy here is a small randomly-initialized MLP,
not a trained locomotion policy. A trained policy produces sustained, regular
contact; a random one produces a different contact pattern, so these numbers
are indicative rather than final. The spike exists to choose a simulator and a
horizon, not to produce a publishable number. Re-run C1-C4 against the real
policy at all -- that is the actual exit criterion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from imu_diff_sim import noise  # noqa: E402
from imu_diff_sim.params import (  # noqa: E402
    UNCONSTRAINED_FIELDS,
    bmi270_nominal,
    from_unconstrained,
    to_unconstrained,
)

TARGET_HEIGHT = 0.55  # ant torso height when standing, metres


def make_policy(key, obs_size: int, act_size: int, hidden: int = 32):
    """Small fixed MLP. Weights are small so initial actions are modest."""
    k1, k2, k3 = jax.random.split(key, 3)
    s = 0.3
    return {
        "w1": jax.random.normal(k1, (obs_size, hidden)) * s / np.sqrt(obs_size),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, hidden)) * s / np.sqrt(hidden),
        "b2": jnp.zeros(hidden),
        "w3": jax.random.normal(k3, (hidden, act_size)) * s / np.sqrt(hidden),
        "b3": jnp.zeros(act_size),
    }


def apply_policy(theta, obs):
    h = jnp.tanh(obs @ theta["w1"] + theta["b1"])
    h = jnp.tanh(h @ theta["w2"] + theta["b2"])
    return jnp.tanh(h @ theta["w3"] + theta["b3"])


def build_loss(env, theta, gyro_slice, horizon, dt):
    """loss(u, eps) -> scalar, differentiable in u.

    The IMU enters where it physically would: the policy reads a corrupted
    angular velocity. Only the gyro is injected -- one channel is enough to
    test whether gradients survive the unroll, and it avoids needing a
    finite-differenced linear acceleration that would itself be noisy.

    The loss is mean squared torso-height deviation: smooth, and it grows when
    the robot falls. Deliberately NOT the env reward, which contains a boolean
    'healthy' term -- a step function has zero gradient almost everywhere and
    an undefined one at the threshold.
    """
    lo, hi = gyro_slice

    def loss(u, eps):
        p = from_unconstrained(u)

        def step(carry, eps_k):
            state, bias = carry
            true_imu = noise.ImuSample(
                gyroscope=state.pipeline_state.xd.ang[0],
                accelerometer=jnp.zeros(3),
            )
            bias, meas = noise.step(bias, true_imu, p, eps_k, dt)
            residual = meas.gyroscope - true_imu.gyroscope
            obs = state.obs.at[lo:hi].add(residual)
            state = env.step(state, apply_policy(theta, obs))
            height = state.pipeline_state.x.pos[0, 2]
            return (state, bias), (height - TARGET_HEIGHT) ** 2

        state0 = env.reset(jax.random.PRNGKey(0))
        bias0 = noise.initial_bias(p)
        _, sq = jax.lax.scan(step, (state0, bias0), eps)
        return jnp.mean(sq)

    return loss


def analyse(name, loss_fn, u0, horizon, n_draws, fd_step):
    """Evaluate C1-C4 for one (backend, horizon).

    Both the loss and its gradient are jitted once and reused across noise
    draws. Without this every call re-traces the scan, which dominates the
    runtime and makes the long horizons impractical. Compilation cost is
    independent of H because scan compiles its body once.
    """
    loss_j = jax.jit(loss_fn)
    grad_j = jax.jit(jax.grad(loss_fn))

    # How much does the loss actually move across the search box? If it barely
    # moves, there is nothing for ANY search method to find and the three-arm
    # comparison is vacuous -- a far more important thing to learn now than
    # whether the gradient is well conditioned. u = +-3 is effectively the box
    # edge, since tanh(3) = 0.995.
    eps_s = noise.draw_noise(jax.random.PRNGKey(1000), (horizon,))
    l_lo = float(loss_j(jnp.full_like(u0, -3.0), eps_s))
    l_mid = float(loss_j(u0, eps_s))
    l_hi = float(loss_j(jnp.full_like(u0, 3.0), eps_s))
    box_span = abs(l_hi - l_lo) / max(abs(l_mid), 1e-30)

    grads = []
    for d in range(n_draws):
        eps = noise.draw_noise(jax.random.PRNGKey(1000 + d), (horizon,))
        grads.append(np.asarray(grad_j(u0, eps)))
    G = np.stack(grads)

    c1 = bool(np.all(np.isfinite(G)))

    # C2: central finite differences on draw 0
    eps0 = noise.draw_noise(jax.random.PRNGKey(1000), (horizon,))
    rel_errs = []
    for i in range(u0.shape[0]):
        e = jnp.zeros_like(u0).at[i].set(fd_step)
        num = float((loss_j(u0 + e, eps0) - loss_j(u0 - e, eps0)) / (2 * fd_step))
        ana = float(G[0, i])
        denom = max(abs(num), abs(ana), 1e-30)
        rel_errs.append(abs(ana - num) / denom)
    # Restrict to components that actually influence the loss; a zero
    # analytic derivative compared against finite-difference rounding noise
    # produces a meaningless 100% "error".
    scale0 = np.max(np.abs(G[0])) if np.max(np.abs(G[0])) > 0 else 1.0
    act0 = np.abs(G[0]) > 1e-9 * scale0
    sel = [e for e, a in zip(rel_errs, act0) if a] or rel_errs
    med_rel = float(np.median(sel))
    c2 = med_rel < 0.20

    # C3: sign agreement with the majority, over ACTIVE components only.
    #
    # A parameter whose gradient is identically zero is not an unstable
    # gradient -- it is a parameter this setup cannot see. On the brax ant the
    # observation carries angular velocity but no linear acceleration, so the
    # three accelerometer parameters are structurally inactive and their
    # sign is 0. Counting those as disagreements caps the metric at 0.5 and
    # makes a healthy gradient look chaotic. (On a Go1-style task the
    # accelerometer enters via projected gravity and becomes active.)
    scale = np.max(np.abs(G)) if np.max(np.abs(G)) > 0 else 1.0
    active = np.median(np.abs(G), axis=0) > 1e-9 * scale
    signs = np.sign(G[:, active])
    if signs.size:
        majority = np.sign(np.sum(signs, axis=0))
        majority[majority == 0] = 1.0
        per_comp = np.mean(signs == majority[None, :], axis=0)
        agree = float(np.mean(per_comp))
    else:
        per_comp = np.array([])
        agree = float("nan")
    # With very few draws the majority sign is ill-defined and `agree`
    # degenerates toward 0.5 regardless of the gradient's real stability.
    c3 = bool(agree > 0.8) if n_draws >= 6 else False

    norm = float(np.median(np.linalg.norm(G, axis=1)))
    return {
        "name": name, "H": horizon, "C1": c1, "C2": c2, "C3": c3,
        "rel_err": med_rel, "agree": agree, "norm": norm,
        "grad_mean": G.mean(axis=0), "active": active, "per_comp": per_comp,
        "n_active": int(active.sum()), "l_lo": l_lo, "l_mid": l_mid,
        "l_hi": l_hi, "box_span": box_span,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="ant")
    ap.add_argument("--backends", nargs="+", default=["positional", "mjx"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[50, 100, 200, 400])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fd-step", type=float, default=1e-3)
    args = ap.parse_args()

    from brax import envs  # imported late so --help works without brax

    all_rows = []
    for backend in args.backends:
        print(f"\n{'='*78}\nBACKEND: {backend}\n{'='*78}")
        try:
            env = envs.create(args.env, backend=backend, episode_length=10_000)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not create env: {type(exc).__name__}: {exc}")
            continue

        dt = float(env.dt)
        state0 = env.reset(jax.random.PRNGKey(0))
        obs_size = int(state0.obs.shape[0])
        q, qd = int(env.sys.q_size()), int(env.sys.qd_size())
        # brax locomotion obs = concat(qpos[2:], qvel); the free joint's angular
        # velocity is qvel[3:6], so it lands here. Printed so it is auditable.
        lo = (q - 2) + 3
        hi = lo + 3
        print(f"  dt={dt:.4f}s  obs={obs_size}  q={q} qd={qd}  gyro obs slice=[{lo}:{hi}]")
        if hi > obs_size:
            print("  gyro slice outside obs; skipping backend")
            continue

        theta = make_policy(jax.random.PRNGKey(7), obs_size, int(env.action_size))
        u0 = to_unconstrained(bmi270_nominal())

        for H in args.horizons:
            loss_fn = build_loss(env, theta, (lo, hi), H, dt)
            t0 = time.time()
            try:
                row = analyse(f"{backend}", loss_fn, u0, H, args.draws, args.fd_step)
            except Exception as exc:  # noqa: BLE001
                print(f"  H={H:<4} FAILED: {type(exc).__name__}: {exc}")
                continue
            row["secs"] = time.time() - t0
            all_rows.append(row)
            print(
                f"  H={H:<4} C1={'ok' if row['C1'] else 'FAIL'} "
                f"C2={'ok' if row['C2'] else 'FAIL'}(rel {row['rel_err']:.1%}) "
                f"C3={'ok' if row['C3'] else 'FAIL'}(agree {row['agree']:.2f}, "
                f"{row['n_active']}/{len(row['grad_mean'])} active) "
                f"||g||={row['norm']:.3e}  [{row['secs']:.0f}s]"
            )
            pc = ", ".join(
                f"{n.replace('scope','').replace('erometer','')}={v:.2f}"
                for n, v in zip(
                    [f for f, a in zip(UNCONSTRAINED_FIELDS, row['active']) if a],
                    row['per_comp'],
                )
            )
            print(f"        per-component sign agreement: {pc}")
            print(
                f"        loss across box: lo={row['l_lo']:.4g} "
                f"mid={row['l_mid']:.4g} hi={row['l_hi']:.4g} "
                f"-> span {row['box_span']:.1%} of mid"
            )

    # C4 needs the whole horizon sweep per backend
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'backend':<12}{'H':>6}{'C1':>5}{'C2':>5}{'C3':>5}{'C4':>5}"
          f"{'rel err':>10}{'sign':>7}{'||g||':>12}")
    print("-" * 78)
    for backend in {r["name"] for r in all_rows}:
        rows = sorted([r for r in all_rows if r["name"] == backend], key=lambda r: r["H"])
        norms = np.array([r["norm"] for r in rows])
        Hs = np.array([r["H"] for r in rows], dtype=float)
        # C4: fit ||g|| ~ H^k in log-log. k <= 1.5 is acceptable; chaotic
        # unrolls blow up far faster than linearly.
        if len(rows) >= 2 and np.all(norms > 0):
            k = float(np.polyfit(np.log(Hs), np.log(norms), 1)[0])
        else:
            k = float("nan")
        c4 = bool(np.isfinite(k) and k <= 1.5)
        for r in rows:
            print(
                f"{r['name']:<12}{r['H']:>6}{'ok' if r['C1'] else 'FAIL':>5}"
                f"{'ok' if r['C2'] else 'FAIL':>5}{'ok' if r['C3'] else 'FAIL':>5}"
                f"{'ok' if c4 else 'FAIL':>5}{r['rel_err']:>9.1%}{r['agree']:>7.2f}"
                f"{r['norm']:>12.3e}"
            )
        print(f"{'':<12}  ||g|| ~ H^{k:.2f}  -> C4 {'ok' if c4 else 'FAIL'}\n")

        passing = [r["H"] for r in rows if r["C1"] and r["C2"] and r["C3"] and c4]
        if passing:
            print(f"  {backend}: longest fully-passing horizon H = {max(passing)}")
        else:
            print(f"  {backend}: NO horizon passes all four criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
