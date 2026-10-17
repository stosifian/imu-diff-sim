"""Allan analysis of a REAL IMU log, with pre-flight diagnostics.

Allan variance measures a sensor's noise only if the record actually satisfies
its assumptions: the device is stationary, the sampling is uniform, and the
stream is not being filtered or de-biased by firmware. Violate any of those and
the estimator still returns a smooth, plausible, wrong curve. So this script
refuses to report coefficients until it has checked.

Diagnostics run before any fit:

  1. STATIONARITY. Allan variance on a moving device measures the motion.
     This is not a subtle contamination -- it dominates completely.
  2. SAMPLE UNIFORMITY. Jitter and dropped samples corrupt the short-tau end,
     where tau is assumed to be exactly m*dt.
  3. PSD SHAPE. A high-pass corner at very low frequency means the stream is
     being de-biased online (Android's calibrated gyro, iOS deviceMotion).
     That erases the long-tau regime -- the part an overnight log exists to
     capture -- while leaving a curve that looks fine.
  4. BIAS BEHAVIOUR. A raw MEMS bias random-walks over hours. A de-biased
     stream stays pinned near zero and mean-reverts.

Formats: `simple` (timestamp_s,gx,gy,gz,ax,ay,az) and `euroc`
(timestamp_ns,wx,wy,wz,ax,ay,az -- the mav0/imu0/data.csv layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from imu_diff_sim import allan  # noqa: E402
from imu_diff_sim.params import bmi270_nominal  # noqa: E402

GYRO_STATIONARY_TOL = 0.02   # rad/s; peak |omega| above this is motion, not noise
JITTER_TOL = 0.05            # fractional std of sample interval
MIN_HOURS_FOR_RANDOM_WALK = 2.0


def load(path: Path, fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_seconds, gyro Nx3 [rad/s], accel Nx3 [m/s^2])."""
    raw = np.genfromtxt(path, delimiter=",", comments="#", skip_header=0)
    if raw.ndim == 1:
        raise SystemExit(f"{path}: could not parse as CSV with >=7 columns")
    if np.isnan(raw[0]).any():          # header row survived
        raw = raw[1:]
    if fmt == "auto":
        fmt = "euroc" if raw[0, 0] > 1e15 else "simple"
        print(f"[format] detected: {fmt}")
    t = raw[:, 0] / 1e9 if fmt == "euroc" else raw[:, 0]
    return t - t[0], raw[:, 1:4], raw[:, 4:7]


def diagnostics(t, gyro, accel) -> list[str]:
    """Run the pre-flight checks. Returns a list of blocking problems."""
    problems: list[str] = []
    n = len(t)
    dur = t[-1] - t[0]
    dt = dur / (n - 1)
    print(f"\n{'='*74}\nDIAGNOSTICS\n{'='*74}")
    print(f"samples        {n:,}")
    print(f"duration       {dur:.1f} s  ({dur/3600:.2f} h)")
    print(f"mean interval  {dt*1e3:.3f} ms  ({1/dt:.1f} Hz)")

    # 1. stationarity
    gmag = np.linalg.norm(gyro, axis=1)
    peak, rms = float(gmag.max()), float(np.sqrt(np.mean(gmag**2)))
    print(f"\n[1] stationarity: peak |omega| {peak:.4f} rad/s, rms {rms:.4f} rad/s")
    if peak > GYRO_STATIONARY_TOL:
        problems.append(
            f"NOT STATIONARY: peak |omega| = {peak:.3f} rad/s exceeds "
            f"{GYRO_STATIONARY_TOL} rad/s. Allan variance of this record measures "
            f"the motion of the device, not the noise of the sensor."
        )
        print("    FAIL -- device was moving")
    else:
        print("    pass")

    # 2. sample uniformity
    d = np.diff(t)
    jit = float(np.std(d) / np.mean(d))
    drops = int(np.sum(d > 3 * np.median(d)))
    print(f"\n[2] uniformity: interval std/mean {jit:.4f}, {drops} gaps > 3x median")
    if jit > JITTER_TOL:
        problems.append(f"SAMPLING JITTER {jit:.1%} corrupts the short-tau end.")
        print("    FAIL -- excessive jitter")
    else:
        print("    pass")

    # 3. PSD shape -- look for a low-frequency high-pass corner
    fs = 1.0 / np.mean(d)
    nper = min(len(gyro), 1 << 16)
    f, pxx = sps.welch(gyro[:, 0] - gyro[:, 0].mean(), fs=fs, nperseg=nper)
    band = (f > 0) & (f < fs / 4)
    lo = (f > 0) & (f <= max(f[f > 0].min() * 10, 0.01))
    hi = (f > 0.1 * fs / 4) & band
    ratio = float(np.median(pxx[lo]) / np.median(pxx[hi])) if lo.any() and hi.any() else np.nan
    print(f"\n[3] PSD low/high band power ratio: {ratio:.3f}")
    print(f"    (white noise -> ~1; << 1 means a high-pass, i.e. online de-biasing;")
    print(f"     >> 1 is normal for raw data with bias drift or motion)")
    if np.isfinite(ratio) and ratio < 0.5:
        problems.append(
            f"LOW-FREQUENCY POWER SUPPRESSED (ratio {ratio:.2f}): the stream looks "
            "de-biased by firmware or OS. The long-tau Allan regime is destroyed; "
            "use an UNCALIBRATED stream."
        )
        print("    FAIL -- looks high-passed")
    else:
        print("    pass")

    # 4. bias behaviour over the record
    k = max(len(gyro) // 20, 1)
    seg = np.array([gyro[i : i + k, 0].mean() for i in range(0, len(gyro) - k + 1, k)])
    drift = float(seg.max() - seg.min())
    print(f"\n[4] gyro-x segment-mean range over record: {drift:.3e} rad/s")
    print("    (raw bias random-walks; a de-biased stream stays pinned near zero)")

    if dur / 3600 < MIN_HOURS_FOR_RANDOM_WALK:
        problems.append(
            f"RECORD TOO SHORT ({dur/3600:.2f} h): the random-walk regime needs "
            f"long tau. default_taus caps at N/10, giving tau_max = {dur/10:.0f} s. "
            f"At least {MIN_HOURS_FOR_RANDOM_WALK} h is wanted for a usable "
            "random-walk fit."
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--format", choices=["auto", "simple", "euroc"], default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="fit and plot even if diagnostics fail (marks the figure)")
    args = ap.parse_args()

    t, gyro, accel = load(args.log, args.format)
    problems = diagnostics(t, gyro, accel)

    print(f"\n{'='*74}")
    if problems:
        print("BLOCKING PROBLEMS\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        if not args.force:
            print("Refusing to report coefficients. Re-run with --force to see the\n"
                  "curve anyway; it will be marked as diagnostics-failed.")
            return 1
        print("--force given: continuing. Coefficients below are NOT trustworthy.")
    else:
        print("All diagnostics passed.")
    print("=" * 74)

    dt = float((t[-1] - t[0]) / (len(t) - 1))
    nominal = bmi270_nominal()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, name, data, unit in (
        (axes[0], "gyroscope", gyro, "rad/s"),
        (axes[1], "accelerometer", accel, "m/s$^2$"),
    ):
        taus, adev, nc = allan.overlapping_allan_deviation(data, dt)
        conf = allan.allan_confidence(adev, nc)
        print(f"\n{name}:")
        for j, axis in enumerate("xyz"):
            nd = allan.fit_white_noise(taus, adev[:, j])
            rw = allan.fit_random_walk(taus, adev[:, j])
            bi = allan.fit_bias_instability(taus, adev[:, j])
            print(f"  {axis}: noise_density {nd:.4g}  random_walk {rw:.4g}  "
                  f"bias_instability {bi:.4g}")
            ax.errorbar(taus, adev[:, j], yerr=conf[:, j], fmt="o-", ms=2.5, lw=0.8,
                        capsize=1.5, alpha=0.85, label=f"measured {axis}")

        tt = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 200)
        nd0 = float(getattr(nominal, f"{name}_noise_density"))
        rw0 = float(getattr(nominal, f"{name}_random_walk"))
        ax.plot(tt, np.sqrt((nd0 / np.sqrt(tt)) ** 2 + (rw0 * np.sqrt(tt / 3)) ** 2),
                "k--", lw=1.4, label="BMI270 model (nominal)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"averaging time $\tau$ [s]")
        ax.set_ylabel(f"Allan deviation [{unit}]")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)

    title = f"Allan deviation: {args.log.name}  ({(t[-1]-t[0])/3600:.2f} h @ {1/dt:.0f} Hz)"
    if problems:
        title += "   [DIAGNOSTICS FAILED -- NOT A VALID CHARACTERIZATION]"
    fig.suptitle(title, fontsize=10, color="crimson" if problems else "black")

    out = args.out or Path("reports/figures") / f"allan-{args.log.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
