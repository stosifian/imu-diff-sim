"""Pipeline self-test: synthetic noise with KNOWN coefficients must come back out.

Every threshold below was fixed in docs/validation_protocol.md (section D)
BEFORE this file was run for the first time. If a check fails, the fix is in
the code or the protocol is amended in writing with the reason -- the
threshold is never quietly loosened until the check passes.

Checks (names match the protocol):
  G1  generator PSD level/slope vs IEEE 952 analytic spectra (scipy Welch)
  C2  ADEV implementation vs allantools and a brute-force loop
  C3  coefficient recovery on a 2e6-sample trace
  C4  null test: no flicker in -> B's interval must include 0
  C5  per-point EDF calibration vs empirical scatter across seeds
  C6  bootstrap interval coverage across seeds            (slow)

Run:  python -m imu_diff_sim.validation.selftest [--slow]
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import jax
import numpy as np

from . import adev_jax
from .synth import NoiseCoeffs, flicker_kasdin, random_walk, synth_trace, white

DT = 0.005  # 200 Hz, the planned sim IMU rate

# Pre-registered thresholds (protocol section D). Plots draw these same bands.
G1_WHITE_TOL = 0.05      # white PSD level, relative
G1_FLICKER_TOL = 0.10    # flicker PSD level, relative
G1_SLOPE_TOL = 0.05      # flicker log-log slope, absolute
G1_FLICKER_BAND = (0.05, 5.0)  # Hz
C2_TOL = 1e-9            # ADEV agreement, relative
C5_TOL = 0.20            # empirical / predicted var(sigma^2), relative
C5_MS = (1, 10, 100, 1000, 2000)
C6_BAND = (0.91, 0.99)   # bootstrap 95% CI coverage


class Check(NamedTuple):
    name: str
    passed: bool
    detail: str


def _trace(seed: int, n: int, coeffs: NoiseCoeffs) -> np.ndarray:
    with jax.enable_x64(True):
        return np.asarray(synth_trace(jax.random.PRNGKey(seed), n, DT, coeffs))


# ---------------------------------------------------------------------------
# G1: the generator, checked WITHOUT any Allan code
# ---------------------------------------------------------------------------

def check_generator_psd() -> list[Check]:
    """One-sided Welch PSD: white -> 2N^2 (+-5%); flicker -> B^2/(pi f) (+-10%),
    log-log slope -1 (+-0.05) over 0.05-5 Hz. Pins the sqrt(2) convention."""
    from scipy.signal import welch

    out = []
    N, B = 1e-3, 1e-3
    with jax.enable_x64(True):
        xw = np.asarray(white(jax.random.PRNGKey(10), 2**20, N, DT))
        xf = np.asarray(flicker_kasdin(jax.random.PRNGKey(11), 2**20, B, DT))

    f, p = welch(xw, fs=1 / DT, nperseg=4096)
    ratio = np.mean(p[1:-1]) / (2 * N**2)
    out.append(Check("G1 white one-sided PSD = 2N^2", abs(ratio - 1) <= G1_WHITE_TOL,
                     f"PSD/(2N^2) = {ratio:.4f} (tol {G1_WHITE_TOL})"))

    f, p = welch(xf, fs=1 / DT, nperseg=2**16)
    band = (f >= G1_FLICKER_BAND[0]) & (f <= G1_FLICKER_BAND[1])
    level = np.exp(np.mean(np.log(p[band] * np.pi * f[band] / B**2)))
    slope = np.polyfit(np.log(f[band]), np.log(p[band]), 1)[0]
    out.append(Check("G1 flicker one-sided PSD = B^2/(pi f)", abs(level - 1) <= G1_FLICKER_TOL,
                     f"geo-mean PSD/(B^2/(pi f)) = {level:.4f} (tol {G1_FLICKER_TOL})"))
    out.append(Check("G1 flicker PSD slope = -1", abs(slope + 1) <= G1_SLOPE_TOL,
                     f"slope = {slope:.4f} (tol {G1_SLOPE_TOL})"))
    return out


# ---------------------------------------------------------------------------
# C2: estimator vs independent references
# ---------------------------------------------------------------------------

def _brute_force_oadev(rate: np.ndarray, dt: float, m: int) -> float:
    """Literal transcription of the IEEE 1554 overlapping estimator, loops."""
    theta = [0.0]
    for r in rate:
        theta.append(theta[-1] + r * dt)
    M, tau, acc = len(theta), m * dt, 0.0
    for k in range(M - 2 * m):
        d = theta[k + 2 * m] - 2 * theta[k + m] + theta[k]
        acc += d * d
    return float(np.sqrt(acc / (2 * tau * tau * (M - 2 * m))))


def check_estimator() -> list[Check]:
    """Matches allantools.oadev to 1e-9 relative; matches brute force at n=2000."""
    import allantools

    out = []
    x = _trace(20, 200_000, NoiseCoeffs(1e-3, 2e-4, 2e-5))
    taus, adev, _ = adev_jax.overlapping_adev(x, DT)
    _, ref, _, _ = allantools.oadev(x, rate=1 / DT, data_type="freq", taus=taus)
    err = float(np.max(np.abs(adev / ref - 1)))
    out.append(Check("C2 vs allantools.oadev", err <= C2_TOL, f"max rel err {err:.2e} (tol {C2_TOL})"))

    xs = x[:2000]
    ms = np.array([1, 3, 10, 50, 200, 999])
    _, adev_s, _ = adev_jax.overlapping_adev(xs, DT, ms)
    ref_s = np.array([_brute_force_oadev(xs, DT, m) for m in ms])
    err = float(np.max(np.abs(adev_s / ref_s - 1)))
    out.append(Check("C2 vs brute-force loop", err <= C2_TOL, f"max rel err {err:.2e} (tol {C2_TOL})"))
    return out


# ---------------------------------------------------------------------------
# C3 / C4: recovery and the null test
# ---------------------------------------------------------------------------

#: BMI270-like white and RW (params.py nominal), with a flicker level chosen so
#: all three regimes are visible in a 10^4 s record: white meets the flicker
#: floor at ~3 s, flicker meets the RW at ~120 s.
RECOVERY_TRUTH = NoiseCoeffs(N=1.116e-4, B=1e-4, K=1.06e-5)
RECOVERY_TOL = NoiseCoeffs(N=0.05, B=0.20, K=0.25)


def check_recovery(n: int = 2_000_000, n_boot: int = 100) -> list[Check]:
    x = _trace(30, n, RECOVERY_TRUTH)
    fit = adev_jax.fit_with_ci(x, DT, n_boot=n_boot, batch_size=1)
    out = []
    for i, name in enumerate("NBK"):
        est, true, tol = fit.coeffs[i], RECOVERY_TRUTH[i], RECOVERY_TOL[i]
        rel = est / true - 1
        inside = fit.lower[i] <= true <= fit.upper[i]
        out.append(Check(
            f"C3 recover {name}", abs(rel) <= tol and inside,
            f"est {est:.4g} true {true:.4g} rel {rel:+.3f} (tol {tol}); "
            f"95% CI [{fit.lower[i]:.4g}, {fit.upper[i]:.4g}] contains truth: {inside}",
        ))
    return out


def check_null_flicker(n: int = 2_000_000, n_boot: int = 100) -> list[Check]:
    """White + RW only. B's 95% interval must reach 0: the fit must be able to
    say "no flicker", not report the white/RW crossover as a bias-instability
    floor."""
    truth = RECOVERY_TRUTH._replace(B=0.0)
    x = _trace(40, n, truth)
    fit = adev_jax.fit_with_ci(x, DT, n_boot=n_boot, batch_size=1)
    return [Check("C4 null flicker: B CI includes 0", fit.lower.B == 0.0,
                  f"B est {fit.coeffs.B:.3g}, 95% CI [{fit.lower.B:.3g}, {fit.upper.B:.3g}]")]


# ---------------------------------------------------------------------------
# C5: per-point EDF calibration
# ---------------------------------------------------------------------------

def edf_ratios(n: int = 20_000, seeds: int = 300) -> dict[str, np.ndarray]:
    """Empirical var(sigma^2)/mean(sigma^2)^2 divided by predicted 2/edf, at
    C5_MS, for each pure noise type. 1.0 means the EDF formula is exact."""
    ms = np.array(C5_MS)
    gens = {
        "white": (adev_jax.WHITE, lambda k: white(k, n, 1.0, DT)),
        "flicker": (adev_jax.FLICKER, lambda k: flicker_kasdin(k, n, 1.0, DT)),
        "RW": (adev_jax.RANDOM_WALK, lambda k: random_walk(k, n, 1.0, DT)),
    }
    out = {}
    for name, (kind, gen) in gens.items():
        with jax.enable_x64(True):
            keys = jax.random.split(jax.random.PRNGKey(50 + kind), seeds)
            xs = np.asarray(jax.vmap(gen)(keys))
        avar = np.stack([adev_jax.overlapping_adev(x, DT, ms)[1] ** 2 for x in xs])
        emp = avar.var(axis=0, ddof=1) / avar.mean(axis=0) ** 2
        with jax.enable_x64(True):
            pred = 2.0 / np.asarray(adev_jax.edf_oadev(n + 1, ms, kind))
        out[name] = emp / pred
    return out


def check_edf(n: int = 20_000, seeds: int = 300) -> list[Check]:
    """For each pure noise type: empirical var(sigma^2)/mean(sigma^2)^2 across
    seeds must be within +-C5_TOL of the predicted 2/edf at C5_MS."""
    return [
        Check(f"C5 EDF calibration, {name}", bool(np.all(np.abs(r - 1) <= C5_TOL)),
              "emp/pred at m=" + ", ".join(f"{m}:{v:.2f}" for m, v in zip(C5_MS, r)))
        for name, r in edf_ratios(n, seeds).items()
    ]


# ---------------------------------------------------------------------------
# C6: bootstrap coverage (slow)
# ---------------------------------------------------------------------------

COVERAGE_TRUTH = NoiseCoeffs(N=1e-3, B=2e-4, K=2.3e-5)


def coverage_intervals(
    n: int = 200_000, seeds: int = 200, n_boot: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(estimate, lower, upper), each (seeds, 3) for (N, B, K)."""
    est, lo, hi = (np.zeros((seeds, 3)) for _ in range(3))
    for s in range(seeds):
        x = _trace(1000 + s, n, COVERAGE_TRUTH)
        fit = adev_jax.fit_with_ci(x, DT, n_boot=n_boot, seed=s)
        est[s], lo[s], hi[s] = fit.coeffs, fit.lower, fit.upper
    return est, lo, hi


def check_coverage(
    n: int = 200_000, seeds: int = 200, n_boot: int = 100, intervals=None
) -> list[Check]:
    """Fraction of 95% bootstrap intervals that contain the truth, over
    `seeds` independent records, must lie in [0.91, 0.99] for each of N, B, K.
    (Binomial(200, 0.95) puts ~99% of its mass in [0.915, 0.98].)"""
    _, lo, hi = intervals if intervals is not None else coverage_intervals(n, seeds, n_boot)
    truth = np.asarray(COVERAGE_TRUTH)
    cov = np.mean((lo <= truth) & (truth <= hi), axis=0)
    return [Check(f"C6 coverage {name}", C6_BAND[0] <= c <= C6_BAND[1], f"{c:.3f} (band {list(C6_BAND)})")
            for name, c in zip("NBK", cov)]


# ---------------------------------------------------------------------------

def run(slow: bool = False) -> list[Check]:
    checks = check_generator_psd() + check_estimator() + check_edf()
    checks += check_recovery() + check_null_flicker()
    if slow:
        checks += check_coverage()
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slow", action="store_true", help="include C6 coverage (~minutes)")
    results = run(slow=ap.parse_args().slow)
    for c in results:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    n_fail = sum(not c.passed for c in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
