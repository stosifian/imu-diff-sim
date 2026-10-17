"""Synthetic self-consistency check for the IMU noise model.

WHAT THIS DOES AND DOES NOT ESTABLISH.

Does: generates a long trace from known parameters, runs the Allan estimator
on it, and recovers those parameters. That verifies the discrete-time scaling
in noise.py (the two sqrt(dt) factors), the estimator in allan.py, and that the
unit conventions agree end to end. It is the check that catches the bugs which
otherwise survive into week 3.

Does NOT: validate the model against reality. Parameters go in, the same
parameters come out -- a correct implementation of a simplified model. The
write-up must call this self-consistency, never validation. External validation
needs a real static log; see scripts/fit_allan.py.

Reference curves for a real, independently-characterised device (the EuRoC
ADIS16448, whose Kalibr parameters ship with that dataset) are overlaid to show
where a consumer part sits relative to a better one. Those are published
parameters, not our measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from a bare checkout without an editable install (matters on Colab, and
# survives a stale .pth file locally).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

jax.config.update("jax_enable_x64", True)  # before any array is created

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imu_diff_sim import allan, noise  # noqa: E402
from imu_diff_sim.params import BMI270_PROVENANCE, bmi270_nominal  # noqa: E402

# Published Kalibr parameters for the EuRoC VI-sensor IMU (ADIS16448), from
# mav0/imu0/sensor.yaml in the EuRoC MAV dataset.
#
# READ THIS BEFORE COMPARING CURVES. These are ESTIMATOR TUNING values, not an
# Allan characterization. It is standard VIO practice to inflate noise
# parameters well above measured values so the filter stays consistent against
# unmodelled effects. The noise densities here are credible against the
# ADIS16448 datasheet (gyro ~2.4e-4 rad/s/sqrt(Hz), accel ~2.3e-3 m/s^2/sqrt(Hz)),
# but the accelerometer random walk of 3.0e-3 m/s^3/sqrt(Hz) is roughly 40x the
# BMI270 figure and is not physically plausible for a part of that grade.
#
# Consequence for the plot: the ADIS16448 reference curve sits ABOVE the BMI270
# curve at long tau. That does NOT mean the consumer part is better. It means
# one curve is a measurement and the other is a conservative tuning choice.
# Do not use this comparison as evidence of anything in the write-up.
ADIS16448 = {
    "gyroscope_noise_density": 1.6968e-04,   # rad/s/sqrt(Hz)
    "gyroscope_random_walk": 1.9393e-05,     # rad/s^2/sqrt(Hz)
    "accelerometer_noise_density": 2.0000e-03,  # m/s^2/sqrt(Hz)
    "accelerometer_random_walk": 3.0000e-03,    # m/s^3/sqrt(Hz)
}


def theoretical_adev(tau: np.ndarray, nd: float, rw: float) -> np.ndarray:
    """Allan deviation of white noise + random walk, in quadrature.

    sigma^2(tau) = (N/sqrt(tau))^2 + (K sqrt(tau/3))^2

    Note what is absent: there is no flicker term, so this curve has no flat
    bias-instability floor -- only a minimum where the two slopes cross. That
    is a property of the model, not of the plot.
    """
    return np.sqrt((nd / np.sqrt(tau)) ** 2 + (rw * np.sqrt(tau / 3.0)) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, default=0.005, help="sampling interval [s]")
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0, help="seed for the plotted trace")
    ap.add_argument(
        "--seeds", type=int, default=16,
        help="independent realizations used for the recovery statistics. "
             "One realization is NOT enough: random-walk recovery has large "
             "single-trace scatter because the long-tau end has few independent "
             "clusters, and a single draw can look like a systematic bias. Fewer than ~16 and the MEAN is itself too noisy to judge bias against.",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("reports/figures/allan-self-consistency.png")
    )
    args = ap.parse_args()

    p = bmi270_nominal()
    duration_h = args.samples * args.dt / 3600.0
    print(
        f"Generating {args.samples:,} samples at dt={args.dt} s "
        f"({duration_h:.2f} h of simulated time), seed={args.seed}"
    )

    eps = noise.draw_noise(jax.random.PRNGKey(args.seed), (args.samples,))
    _, out = noise.corrupt_trace(
        noise.initial_bias(p), noise.zero_trace(args.samples), p, eps, args.dt
    )

    def recover(seed: int):
        """Allan-fit one realization. Returns per-channel (taus, adev, conf, fits)."""
        e = noise.draw_noise(jax.random.PRNGKey(seed), (args.samples,))
        _, o = noise.corrupt_trace(
            noise.initial_bias(p), noise.zero_trace(args.samples), p, e, args.dt
        )
        res = {}
        for chan, sig in (
            ("gyroscope", np.asarray(o.gyroscope)[:, 0]),
            ("accelerometer", np.asarray(o.accelerometer)[:, 0]),
        ):
            t, a, nc = allan.overlapping_allan_deviation(sig, args.dt)
            res[chan] = (
                t, a, allan.allan_confidence(a, nc),
                allan.fit_white_noise(t, a),
                allan.fit_random_walk(t, a),
                allan.fit_bias_instability(t, a),
            )
        return res

    seeds = sorted({args.seed} | set(range(args.seeds)))
    print(f"Running {len(seeds)} realizations: seeds {seeds}")
    results = {sd: recover(sd) for sd in seeds}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    rows = []

    for chan, unit, ax in (
        ("gyroscope", "rad/s", axes[0]),
        ("accelerometer", "m/s$^2$", axes[1]),
    ):
        nd_in = float(getattr(p, f"{chan}_noise_density"))
        rw_in = float(getattr(p, f"{chan}_random_walk"))

        taus, adev, conf, _, _, _ = results[args.seed][chan]
        nd_all = np.array([results[sd][chan][3] for sd in seeds])
        rw_all = np.array([results[sd][chan][4] for sd in seeds])
        bi_all = np.array([results[sd][chan][5] for sd in seeds])

        rows.append((chan, "noise_density", nd_in, nd_all))
        rows.append((chan, "random_walk", rw_in, rw_all))
        rows.append((chan, "bias_instability", float("nan"), bi_all))

        ax.errorbar(
            taus, adev, yerr=conf, fmt="o", ms=3, lw=0.8, capsize=2,
            color="#1f77b4", label="simulated trace", zorder=3,
        )
        tt = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 200)
        ax.plot(tt, theoretical_adev(tt, nd_in, rw_in), "k--", lw=1.2,
                label="model, analytic", zorder=4)
        ax.plot(tt, nd_in / np.sqrt(tt), ":", color="gray", lw=1,
                label=r"white noise, $\tau^{-1/2}$")
        ax.plot(tt, rw_in * np.sqrt(tt / 3.0), "-.", color="gray", lw=1,
                label=r"random walk, $\tau^{+1/2}$")
        ax.plot(
            tt,
            theoretical_adev(tt, ADIS16448[f"{chan}_noise_density"],
                             ADIS16448[f"{chan}_random_walk"]),
            color="#d62728", lw=1.2, alpha=0.75,
            label="ADIS16448 (EuRoC tuning values,\nnot a characterization)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"averaging time $\tau$ [s]")
        ax.set_ylabel(f"Allan deviation [{unit}]")
        ax.set_title(f"{chan} (BMI270 nominal)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"IMU noise model self-consistency  |  {args.samples:,} samples @ "
        f"{1/args.dt:.0f} Hz = {duration_h:.1f} h simulated  |  seed {args.seed}",
        fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nFigure -> {args.out}")

    print(
        f"\n{'channel':<14} {'parameter':<18} {'input':>11} "
        f"{'mean':>11} {'ratio':>8} {'1-sigma':>9} {'z':>7}"
    )
    print("-" * 84)
    worst_bias = 0.0
    for chan, name, a, vals in rows:
        vals = np.asarray(vals, dtype=float)
        mean = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
        sd = float(np.nanstd(vals)) if np.sum(np.isfinite(vals)) > 1 else float("nan")
        ratio = mean / a if (a and np.isfinite(a) and a != 0) else float("nan")
        frac = sd / mean if (np.isfinite(sd) and mean) else float("nan")
        if np.isfinite(ratio):
            worst_bias = max(worst_bias, abs(ratio - 1.0))
        n_ok = int(np.sum(np.isfinite(vals)))
        sem = sd / np.sqrt(n_ok) if (np.isfinite(sd) and n_ok > 1) else float("nan")
        # How many standard errors the mean sits from the input value. Above
        # ~3 and the deviation is real rather than sampling noise.
        z = abs(mean - a) / sem if (np.isfinite(sem) and sem > 0 and np.isfinite(a)) else float("nan")
        print(
            f"{chan:<14} {name:<18} {a:>11.4g} {mean:>11.4g} "
            f"{ratio:>8.3f} {frac:>8.1%} {z:>7.1f}"
        )

    print(
        f"\nWorst mean recovery bias over {len(seeds)} realizations: "
        f"{worst_bias*100:.1f}%"
    )
    print(
        "The 1-sigma column is the spread ACROSS realizations, not an error in the\n"
        "estimator. Random-walk scatter is large because the long-tau end of the\n"
        "curve has only ~10 independent clusters at this record length; a single\n"
        "trace can land 15-20% off and look like a systematic bias. A real static\n"
        "log is one realization, so this is the precision to expect from it too.\n"
        "The z column is |mean - input| in standard errors of the mean: below ~3,\n"
        "the deviation is sampling noise, not a bias in the model or estimator."
    )
    print(
        "\nbias_instability is NaN by design: the model uses a pure random walk, "
        "which contains\nno flicker term, so there is no flat floor to fit. A real "
        "MEMS part usually shows one.\nThat gap is a known limitation of the model, "
        "not a bug in the estimator."
    )
    print(
        "\nREMINDER: this is self-consistency, not validation. "
        "See BMI270_PROVENANCE for\nwhere the input values come from; "
        f"{sum('UNKNOWN' in v for v in BMI270_PROVENANCE.values())} entries are "
        "still tagged UNKNOWN."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
