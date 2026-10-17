"""Diagnostic figures for docs/validation_protocol.md.

Two kinds of figure, kept visibly distinct:

  SELF-TEST (01-07)  real results of the pipeline self-test on synthetic data,
                     with the pre-registered pass/fail bands drawn in.
  SPEC / MODEL (10-17)  what the model predicts and what the BMI270 datasheet
                     (BST-BMI270-DS000-08) allows, with each hardware test's
                     acceptance region drawn. NO MEASURED DATA EXISTS YET; these
                     carry a watermark saying so and are the axes the
                     measurements will be plotted onto.

This lives in scripts/ rather than the validation package because it imports
imu_diff_sim.allan (for the cross-check panel) and params.py, and the package
itself must stay independent of both.

Run:  PYTHONPATH=src python scripts/plot_validation.py [--only 03,10]
Out:  reports/figures/validation/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from imu_diff_sim import params  # noqa: E402
from imu_diff_sim.validation import adev_jax, selftest  # noqa: E402
from imu_diff_sim.validation import bmi270_spec as ds  # noqa: E402
from imu_diff_sim.validation.synth import (  # noqa: E402
    NoiseCoeffs, flicker_kasdin, synth_trace, white,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "validation"
CACHE = OUT / "_cache"
DT = selftest.DT

# Open protocol parameters (placeholders until the sim/task spike lands).
T_EP = 20.0          # episode length [s]
F_POL = 50.0         # policy rate [Hz]
LOG_HOURS = 8.0      # planned static log length
BODY_RATE = 2.0      # assumed peak locomotion body rate [rad/s] for the budget

C_PASS, C_FAIL, C_BAND = "#1a7f37", "#cf222e", "#d0e8d5"
C_MODEL, C_SPEC, C_LUX, C_TRUTH = "#0550ae", "#bc4c00", "#8250df", "#24292f"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10,
    "legend.fontsize": 7.5, "legend.framealpha": 0.9,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _vcolor(ok: bool) -> str:
    return C_PASS if ok else C_FAIL


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def _watermark(fig):
    fig.text(0.5, 0.5, "NO MEASURED DATA YET\nmodel + datasheet only",
             ha="center", va="center", fontsize=26, color="0.85", alpha=0.55,
             rotation=20, zorder=0, weight="bold")


def _sigma(taus, N, B, K):
    """Analytic ADEV, IEEE 952 three-term model."""
    return np.sqrt(N**2 / taus + adev_jax.FLICKER_FACTOR_SQ * B**2 + K**2 * taus / 3.0)


def _trace(seed, n, coeffs):
    with jax.enable_x64(True):
        return np.asarray(synth_trace(jax.random.PRNGKey(seed), n, DT, coeffs))


def _point_ci(taus, n_samples, N, B, K, conf=0.95):
    """Per-point chi^2 interval for a record of n_samples under (N, B, K)."""
    ms = np.round(taus / DT)
    parts = np.stack([N**2 / taus, adev_jax.FLICKER_FACTOR_SQ * B**2 * np.ones_like(taus),
                      K**2 * taus / 3.0], axis=1)
    with jax.enable_x64(True):
        edf = np.asarray(adev_jax.edf_oadev(n_samples + 1, ms, np.argmax(parts, axis=1)))
    return adev_jax.adev_ci(_sigma(taus, N, B, K), edf, conf)


def _adev_panel(ax, fit, truth, title):
    """ADEV points with chi^2 bars, truth, fit and its components."""
    lo, hi = adev_jax.adev_ci(fit.adev, fit.edf)
    ax.errorbar(fit.taus, fit.adev, yerr=[fit.adev - lo, hi - fit.adev], fmt="o", ms=3,
                color=C_MODEL, ecolor="0.6", elinewidth=0.8, capsize=0,
                label="estimate ± 95% χ² (EDF)")
    t = np.logspace(np.log10(fit.taus[0]), np.log10(fit.taus[-1]), 200)
    ax.plot(t, _sigma(t, *truth), color=C_TRUTH, lw=2, alpha=0.5, label="truth (analytic)")
    c = fit.coeffs
    ax.plot(t, _sigma(t, *c), color=C_MODEL, lw=1, label="fit total")
    for val, comp, ls in [(c.N, "N/√τ", ":"), (c.B, "0.664·B", "--"), (c.K, "K√(τ/3)", "-.")]:
        if val > 0:
            sel = {"N/√τ": (val, 0, 0), "0.664·B": (0, val, 0), "K√(τ/3)": (0, 0, val)}[comp]
            ax.plot(t, _sigma(t, *sel), color=C_MODEL, lw=0.8, ls=ls, alpha=0.7,
                    label=f"fit {comp}")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("τ [s]"), ax.set_ylabel("σ(τ) [rad/s]")
    ax.set_title(title)
    ax.legend(loc="lower left")


def _boot_hist(ax, boot, est, lo, hi, truth, tol, name, unit=""):
    ax.hist(boot, bins=25, color="0.75", edgecolor="white")
    if tol is not None and truth > 0:
        ax.axvspan(truth * (1 - tol), truth * (1 + tol), color=C_BAND, alpha=0.6,
                   label=f"tolerance ±{tol:.0%}")
    ax.axvline(truth, color=C_TRUTH, lw=2, label="truth")
    ax.axvline(est, color=C_MODEL, lw=1.2, label="estimate")
    for v in (lo, hi):
        ax.axvline(v, color=C_MODEL, lw=1, ls="--")
    ax.plot([], [], color=C_MODEL, ls="--", label="95% bootstrap CI")
    ax.set_title(name)
    ax.set_xlabel(unit)
    ax.set_yticks([])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper right", fontsize=6.5)


# ---------------------------------------------------------------------------
# SELF-TEST FIGURES
# ---------------------------------------------------------------------------

def fig01_generator_psd():
    N, B = 1e-3, 1e-3
    with jax.enable_x64(True):
        xw = np.asarray(white(jax.random.PRNGKey(10), 2**20, N, DT))
        xf = np.asarray(flicker_kasdin(jax.random.PRNGKey(11), 2**20, B, DT))
    checks = {c.name: c for c in selftest.check_generator_psd()}

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    f, p = welch(xw, fs=1 / DT, nperseg=4096)
    tol = selftest.G1_WHITE_TOL
    a.loglog(f[1:], p[1:], color="0.55", lw=0.6, label="Welch PSD (one-sided)")
    a.axhspan(2 * N**2 * (1 - tol), 2 * N**2 * (1 + tol), color=C_BAND, alpha=0.8,
              label=f"pass band 2N² ±{tol:.0%}")
    a.axhline(2 * N**2, color=C_TRUTH, lw=1.5, label="2N²  (IEEE 952, one-sided)")
    a.axhline(N**2, color=C_FAIL, lw=1, ls=":", label="N²  (what a √2 convention bug would give)")
    c = checks["G1 white one-sided PSD = 2N^2"]
    a.set_title(f"G1 white  [{_verdict(c.passed)}]  {c.detail}", color=_vcolor(c.passed))
    a.set_xlabel("f [Hz]"), a.set_ylabel("PSD [(rad/s)²/Hz]")
    a.legend(loc="lower left")

    f, p = welch(xf, fs=1 / DT, nperseg=2**16)
    lo_f, hi_f = selftest.G1_FLICKER_BAND
    tol = selftest.G1_FLICKER_TOL
    ff = np.logspace(np.log10(lo_f), np.log10(hi_f), 50)
    b.loglog(f[1:], p[1:], color="0.55", lw=0.6, label="Welch PSD (one-sided)")
    b.fill_between(ff, B**2 / (np.pi * ff) * (1 - tol), B**2 / (np.pi * ff) * (1 + tol),
                   color=C_BAND, alpha=0.9, label=f"pass band ±{tol:.0%}, slope −1±{selftest.G1_SLOPE_TOL}")
    ft = f[1:]
    b.loglog(ft, B**2 / (np.pi * ft), color=C_TRUTH, lw=1.2, label="B²/(πf)  (IEEE 952)")
    b.axvspan(lo_f, hi_f, color="0.9", alpha=0.4, zorder=0, label="evaluation band")
    lvl = checks["G1 flicker one-sided PSD = B^2/(pi f)"]
    slp = checks["G1 flicker PSD slope = -1"]
    ok = lvl.passed and slp.passed
    b.set_title(f"G1 flicker (Kasdin–Walter)  [{_verdict(ok)}]\n{lvl.detail};  {slp.detail}",
                color=_vcolor(ok))
    b.set_xlabel("f [Hz]")
    b.legend(loc="lower left")
    fig.suptitle("01  Generator vs analytic spectra — checks the synthetic source WITHOUT any Allan code\n"
                 "(pass bands apply to the band-averaged level, not to individual PSD bins)",
                 weight="bold", y=1.08)
    return _save(fig, "01_g1_generator_psd.png")


def fig02_estimator_agreement():
    import allantools

    from imu_diff_sim import allan

    x = _trace(20, 200_000, NoiseCoeffs(1e-3, 2e-4, 2e-5))
    taus, ad, _ = adev_jax.overlapping_adev(x, DT)
    _, ref, _, _ = allantools.oadev(x, rate=1 / DT, data_type="freq", taus=taus)
    _, ad_main, _ = allan.overlapping_allan_deviation(x, DT, taus)

    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.loglog(taus, ref, "o", ms=7, mfc="none", color=C_SPEC, label="allantools.oadev (reference)")
    a.loglog(taus, ad_main, "s", ms=4, mfc="none", color=C_LUX, label="imu_diff_sim.allan (model session)")
    a.loglog(taus, ad, ".", ms=4, color=C_MODEL, label="validation.adev_jax (this check)")
    a.set_ylabel("σ(τ) [rad/s]")
    a.legend()
    e1 = np.abs(ad / ref - 1)
    e2 = np.abs(ad / ad_main - 1)
    eps = np.finfo(float).eps
    b.semilogy(taus, np.maximum(e1, eps / 10), "o-", ms=3, color=C_SPEC, label="|jax/allantools − 1|")
    b.semilogy(taus, np.maximum(e2, eps / 10), "s-", ms=3, color=C_LUX, label="|jax/allan.py − 1|")
    b.axhline(selftest.C2_TOL, color=C_FAIL, lw=1.2, ls="--", label=f"fail above {selftest.C2_TOL:g}")
    b.axhline(eps, color="0.5", lw=0.8, ls=":", label="float64 ε")
    b.set_xscale("log")
    b.set_xlabel("τ [s]"), b.set_ylabel("relative difference")
    b.legend(loc="center right", fontsize=6.5)
    ok = max(e1.max(), e2.max()) <= selftest.C2_TOL
    a.set_title(f"02  C2 estimator agreement, three independent implementations  [{_verdict(ok)}]\n"
                f"max rel diff: allantools {e1.max():.1e}, allan.py {e2.max():.1e}",
                color=_vcolor(ok), weight="bold")
    return _save(fig, "02_c2_estimator_agreement.png")


def fig03_recovery():
    truth, tol = selftest.RECOVERY_TRUTH, selftest.RECOVERY_TOL
    x = _trace(30, 2_000_000, truth)
    fit = adev_jax.fit_with_ci(x, DT, n_boot=100, batch_size=1)

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[:, 0])
    _adev_panel(ax, fit, truth, "ADEV of a 2e6-sample (10⁴ s) synthetic trace")
    oks = []
    for i, name in enumerate("NBK"):
        h = fig.add_subplot(gs[i, 1])
        rel = fit.coeffs[i] / truth[i] - 1
        ok = abs(rel) <= tol[i] and fit.lower[i] <= truth[i] <= fit.upper[i]
        oks.append(ok)
        _boot_hist(h, fit.boot[:, i], fit.coeffs[i], fit.lower[i], fit.upper[i], truth[i], tol[i],
                   f"{name}: {rel:+.1%} vs truth  [{_verdict(ok)}]")
        h.title.set_color(_vcolor(ok))
    fig.suptitle(f"03  C3 coefficient recovery  [{_verdict(all(oks))}]   "
                 f"truth N={truth.N:.3g}, B={truth.B:.3g}, K={truth.K:.3g}",
                 weight="bold", color=_vcolor(all(oks)))
    fig.tight_layout()
    return _save(fig, "03_c3_recovery.png")


def fig04_null_flicker():
    truth = selftest.RECOVERY_TRUTH._replace(B=0.0)
    x = _trace(40, 2_000_000, truth)
    fit = adev_jax.fit_with_ci(x, DT, n_boot=100, batch_size=1)
    ok = fit.lower.B == 0.0

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    _adev_panel(a, fit, truth, "white + RW only — the curve still has a minimum")
    tmin = fit.taus[np.argmin(fit.adev)]
    a.annotate(f"argmin at τ≈{tmin:.0f} s:\nnaive B = min/0.664 = {fit.adev.min() / 0.664:.2g}\n(a fake bias instability)",
               xy=(tmin, fit.adev.min()), xytext=(0.55, 0.75), textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", color=C_FAIL), color=C_FAIL, fontsize=8)
    frac0 = np.mean(fit.boot[:, 1] == 0.0)
    b.hist(fit.boot[:, 1], bins=25, color="0.75", edgecolor="white")
    b.axvline(0, color=C_TRUTH, lw=2, label="truth B = 0")
    b.axvline(fit.coeffs.B, color=C_MODEL, lw=1.2, label=f"estimate {fit.coeffs.B:.2g}")
    b.axvline(fit.upper.B, color=C_MODEL, lw=1, ls="--", label=f"95% CI [{fit.lower.B:.2g}, {fit.upper.B:.2g}]")
    b.axvline(fit.adev.min() / 0.664, color=C_FAIL, lw=1, ls=":", label="naive argmin B")
    b.set_title(f"bootstrap B: {frac0:.0%} of replicates exactly 0")
    b.set_yticks([])
    b.legend(fontsize=7)
    fig.suptitle(f"04  C4 null test: no flicker in → B's interval must reach 0  [{_verdict(ok)}]",
                 weight="bold", color=_vcolor(ok))
    fig.tight_layout()
    return _save(fig, "04_c4_null_flicker.png")


def fig05_edf():
    ratios = selftest.edf_ratios()
    ms = np.array(selftest.C5_MS)
    tol = selftest.C5_TOL
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(1 - tol, 1 + tol, color=C_BAND, alpha=0.8, label=f"pass band ±{tol:.0%}")
    ax.axhline(1, color=C_TRUTH, lw=1)
    oks = []
    for (name, r), mk in zip(ratios.items(), "osD"):
        ok = bool(np.all(np.abs(r - 1) <= tol))
        oks.append(ok)
        ax.semilogx(ms, r, mk + "-", ms=5, label=f"{name} [{_verdict(ok)}]")
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("cluster size m  (τ = m·dt)")
    ax.set_ylabel("empirical var(σ̂²)/σ⁴  ÷  predicted 2/EDF")
    ax.legend()
    ax.set_title(f"05  C5 per-point error bars: Howe/Allan/Barnes EDF vs 300 seeds  "
                 f"[{_verdict(all(oks))}]\n(n = 20 000 samples per record)",
                 weight="bold", color=_vcolor(all(oks)))
    return _save(fig, "05_c5_edf_calibration.png")


def fig06_coverage():
    path = CACHE / "c6_intervals.npz"
    if path.exists():
        z = np.load(path)
        est, lo, hi, truth = z["est"], z["lo"], z["hi"], z["truth"]
    else:
        est, lo, hi = selftest.coverage_intervals()
        truth = np.asarray(selftest.COVERAGE_TRUTH)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, est=est, lo=lo, hi=hi, truth=truth)
    band = selftest.C6_BAND
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    oks = []
    for i, (ax, name) in enumerate(zip(axs, "NBK")):
        order = np.argsort(est[:, i])
        hit = (lo[:, i] <= truth[i]) & (truth[i] <= hi[:, i])
        cov = hit.mean()
        ok = band[0] <= cov <= band[1]
        oks.append(ok)
        idx = np.arange(len(order))
        for j, k in enumerate(order):
            ax.plot([j, j], [lo[k, i], hi[k, i]], color=C_MODEL if hit[k] else C_FAIL,
                    lw=0.8 if hit[k] else 1.4, alpha=0.6 if hit[k] else 1)
        ax.plot(idx, est[order, i], ".", ms=2, color="k")
        ax.axhline(truth[i], color=C_TRUTH, lw=1.5)
        ax.set_title(f"{name}: coverage {cov:.3f} (band {band[0]}–{band[1]})  [{_verdict(ok)}]\n"
                     f"{(~hit).sum()} misses (red) of {len(hit)}", color=_vcolor(ok))
        ax.set_xlabel("record (sorted by estimate)")
        ax.set_xticks([])
    axs[0].set_ylabel("95% bootstrap interval")
    fig.suptitle(f"06  C6 interval coverage over 200 independent records  [{_verdict(all(oks))}]   "
                 f"truth N={truth[0]:.3g}, B={truth[1]:.3g}, K={truth[2]:.3g}",
                 weight="bold", color=_vcolor(all(oks)))
    fig.tight_layout()
    return _save(fig, "06_c6_coverage.png")


def fig07_mutations():
    import jax.numpy as jnp

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    # (a) sqrt(2) planted in the white-noise density
    N = 1e-3
    with jax.enable_x64(True):
        xw = np.asarray(white(jax.random.PRNGKey(10), 2**20, N * np.sqrt(2), DT))
    f, p = welch(xw, fs=1 / DT, nperseg=4096)
    tol = selftest.G1_WHITE_TOL
    ratio = np.mean(p[1:-1]) / (2 * N**2)
    a.loglog(f[1:], p[1:], color="0.55", lw=0.6, label="PSD of mutated generator")
    a.axhspan(2 * N**2 * (1 - tol), 2 * N**2 * (1 + tol), color=C_BAND, alpha=0.8, label="G1 pass band")
    a.axhline(2 * N**2, color=C_TRUTH, lw=1.2, label="2N² expected")
    caught = abs(ratio - 1) > tol
    a.set_title(f"planted √2 in density → PSD/(2N²) = {ratio:.2f}\n"
                f"G1 {'FAILS — bug caught' if caught else 'passes — BUG MISSED'}",
                color=C_PASS if caught else C_FAIL)
    a.set_xlabel("f [Hz]"), a.legend(loc="lower left")

    # (b) RW model with the /3 dropped
    real = adev_jax._design

    def wrong_design(taus):
        return jnp.stack([1.0 / taus, jnp.full_like(taus, adev_jax.FLICKER_FACTOR_SQ), taus], axis=1)

    truth, ktol = selftest.RECOVERY_TRUTH, selftest.RECOVERY_TOL.K
    x = _trace(30, 2_000_000, truth)
    adev_jax._design = wrong_design
    adev_jax._compiled.cache_clear()
    try:
        fit = adev_jax.fit_with_ci(x, DT, n_boot=40, batch_size=1)
    finally:
        adev_jax._design = real
        adev_jax._compiled.cache_clear()
    rel = fit.coeffs.K / truth.K - 1
    caught = abs(rel) > ktol or not (fit.lower.K <= truth.K <= fit.upper.K)
    _boot_hist(b, fit.boot[:, 2], fit.coeffs.K, fit.lower.K, fit.upper.K, truth.K, ktol,
               f"dropped /3 in K√(τ/3) → K off by {rel:+.0%} (expect 1/√3−1 = −42%)\n"
               f"C3 {'FAILS — bug caught' if caught else 'passes — BUG MISSED'}")
    b.title.set_color(C_PASS if caught else C_FAIL)
    b.set_xlim(0, truth.K * 1.6)
    fig.suptitle("07  Mutation tests: plant the classic convention bugs; the checks must fail",
                 weight="bold")
    fig.tight_layout()
    return _save(fig, "07_mutations.png")


# ---------------------------------------------------------------------------
# SPEC / MODEL FIGURES (no measured data)
# ---------------------------------------------------------------------------

def _nominal(name):
    return float(getattr(params.bmi270_nominal(), name))


def _fig_b1_adev(sensor: str):
    if sensor == "gyro":
        nN, nK, unit = "gyroscope_noise_density", "gyroscope_random_walk", "rad/s"
        ds_lines = [(ds.datasheet_N_gyro("perf"), "datasheet perf. mode, N = 7 mdps/√Hz ÷ √2", "-"),
                    (ds.datasheet_N_gyro("normal"), "datasheet normal mode, N = 10 mdps/√Hz ÷ √2", "--")]
        naive = ds.GYRO_ND_PERF_DPS * ds.DEG
        lsb_q = ds.GYRO_LSB_DPS_2000 * ds.DEG / np.sqrt(12) * np.sqrt(DT)
    else:
        nN, nK, unit = "accelerometer_noise_density", "accelerometer_random_walk", "m/s²"
        ds_lines = [(ds.datasheet_N_accel(), "datasheet, N = 0.16 mg/√Hz ÷ √2", "-")]
        naive = ds.ACC_ND_MG * 1e-3 * ds.G0
        lsb_q = None
    N, K = _nominal(nN), _nominal(nK)
    n_log = int(LOG_HOURS * 3600 / DT)
    taus = adev_jax.default_ms(n_log) * DT
    lo, hi = _point_ci(taus, n_log, N, 0.0, K)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _watermark(fig)
    fN, fK = params.SEARCH_BOX_FACTORS[nN], params.SEARCH_BOX_FACTORS[nK]
    ax.fill_between(taus, _sigma(taus, fN[0] * N, 0, fK[0] * K), _sigma(taus, fN[1] * N, 0, fK[1] * K),
                    color="0.92", label=f"search box (N ×{fN[0]}–{fN[1]}, K ×{fK[0]}–{fK[1]})")
    ax.fill_between(taus, lo, hi, color=C_MODEL, alpha=0.25,
                    label=f"model 95% band for one {LOG_HOURS:.0f} h log (B4 band)")
    ax.loglog(taus, _sigma(taus, N, 0, K), color=C_MODEL, lw=2,
              label=f"model nominal: N={N:.3g}, K={K:.3g}")
    for val, lab, ls in ds_lines:
        ax.loglog(taus, val / np.sqrt(taus), color=C_SPEC, lw=1.3, ls=ls, label=lab + f" = {val:.3g}")
    ax.loglog(taus, naive / np.sqrt(taus), color=C_SPEC, lw=0.8, ls=":",
              label=f"datasheet density read as N (√2 convention error) = {naive:.3g}")
    lux = ds.LUXONIS[nN]
    ax.plot(1.0, lux, "D", color=C_LUX, ms=6, label=f"Luxonis N (1 unit) = {lux:.3g}", zorder=5)
    if lsb_q is not None:
        ax.loglog(taus, lsb_q / np.sqrt(taus), color="0.5", lw=0.8, ls="-.",
                  label="quantization, ±2000 dps range (LSB/√12)")
    # B1 acceptance at tau = 1 s
    ref = ds_lines[0][0]
    ax.errorbar([1.0], [ref], yerr=[[ref - ref / 1.2], [ref * 1.2 - ref]], fmt="none",
                ecolor=C_PASS, elinewidth=6, alpha=0.6, capsize=0, zorder=4,
                label="B1 accept: model N within ×/÷1.2 of measured (shown at datasheet)")
    ratio = N / ref
    ax.axvspan(10 * T_EP, taus[-1], color="0.6", alpha=0.12, zorder=0)
    ax.text(10 * T_EP * 1.1, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e-6,
            f"τ > 10·T_ep = {10 * T_EP:.0f} s:\nsecondary only", fontsize=7, color="0.35")
    ax.set_xlabel("τ [s]"), ax.set_ylabel(f"σ(τ) [{unit}]")
    ax.set_title(f"{'10' if sensor == 'gyro' else '11'}  B1/B4 {sensor} Allan deviation: "
                 f"model vs BMI270 datasheet ({ds.DOC})\n"
                 f"model N / datasheet-implied N (perf. mode) = {ratio:.2f}  "
                 f"→ {'outside' if abs(np.log(ratio)) > np.log(1.2) else 'inside'} the B1 ±20% band"
                 + (f";  vs normal mode = {N / ds_lines[1][0]:.2f}" if len(ds_lines) > 1 else ""),
                 weight="bold", color=C_FAIL if abs(np.log(ratio)) > np.log(1.2) else C_TRUTH)
    ax.legend(loc="lower left", fontsize=6.8)
    return fig


def fig10_b1_gyro():
    return _save(_fig_b1_adev("gyro"), "10_b1_gyro_adev_vs_spec.png")


def fig11_b1_accel():
    return _save(_fig_b1_adev("accel"), "11_b1_accel_adev_vs_spec.png")


def fig12_noise_vs_odr():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    _watermark(fig)
    # gyro
    Ng = _nominal("gyroscope_noise_density") / ds.DEG  # dps/sqrt(Hz), two-sided
    odr, f3 = ds.GYRO_ODR_HZ, ds.GYRO_F3DB_HZ
    a.loglog(odr, ds.GYRO_RMS_PERF_MDPS, "o", color=C_SPEC, label="Table 12 rms, performance mode")
    a.loglog(odr, ds.GYRO_RMS_NORMAL_MDPS, "s", mfc="none", color=C_SPEC, label="Table 11 rms, normal mode")
    a.loglog(odr, 1e3 * ds.GYRO_ND_PERF_DPS * np.sqrt(f3), "-", color=C_SPEC, lw=1,
             label="0.007 dps/√Hz × √f₃dB  (one-sided reading)")
    a.loglog(odr, 1e3 * ds.GYRO_ND_NORMAL_DPS * np.sqrt(f3), "--", color=C_SPEC, lw=1,
             label="0.010 dps/√Hz × √f₃dB  (normal mode, one-sided)")
    a.loglog(odr, 1e3 * ds.GYRO_ND_PERF_DPS * np.sqrt(2 * f3), ":", color="0.4", lw=1,
             label="perf. two-sided reading: 0.007 × √(2·f₃dB)  ✗ misses ●")
    a.loglog(odr, 1e3 * Ng * np.sqrt(2 * f3), "-", color=C_MODEL, lw=2,
             label=f"model nominal N={Ng * ds.DEG:.3g} rad/s/√Hz, band-limited at f₃dB")
    a.loglog(odr, 1e3 * Ng / np.sqrt(1 / odr), "--", color=C_MODEL, lw=1,
             label="model as implemented: per-sample σ = N/√dt (no LPF)")
    a.axvline(1 / DT, color="0.5", lw=0.8, ls=":")
    a.text(1 / DT * 1.05, 25, "sim ODR", fontsize=7, color="0.4")
    a.set_xlabel("ODR [Hz]"), a.set_ylabel("rms noise [mdps]")
    a.set_title("gyro: rms noise vs ODR")
    a.legend(fontsize=6.5)
    # accel
    Na = _nominal("accelerometer_noise_density") / (1e-3 * ds.G0)  # mg/sqrt(Hz)
    odr, f3 = ds.ACC_ODR_HZ, ds.ACC_F3DB_HZ
    b.loglog(odr, ds.ACC_RMS_MG, "o", color=C_SPEC, label="Table 8 rms, normal mode, ±8 g")
    b.loglog(odr, ds.ACC_ND_MG * np.sqrt(f3), "-", color=C_SPEC, lw=1, label="0.16 mg/√Hz × √f₃dB")
    b.loglog(odr, Na * np.sqrt(2 * f3), "-", color=C_MODEL, lw=2,
             label=f"model nominal N={Na * 1e-3 * ds.G0:.3g} m/s²/√Hz, band-limited")
    b.loglog(odr, Na / np.sqrt(1 / odr), "--", color=C_MODEL, lw=1, label="model as implemented (no LPF)")
    b.axvline(1 / DT, color="0.5", lw=0.8, ls=":")
    b.set_xlabel("ODR [Hz]"), b.set_ylabel("rms noise [mg]")
    b.set_title("accel: rms noise vs ODR")
    b.legend(fontsize=6.5)
    sel = ds.GYRO_ODR_HZ <= 1600
    k = (ds.GYRO_RMS_PERF_MDPS / (1e3 * ds.GYRO_ND_PERF_DPS * np.sqrt(ds.GYRO_F3DB_HZ)))[sel]
    kn = (ds.GYRO_RMS_NORMAL_MDPS / (1e3 * ds.GYRO_ND_NORMAL_DPS * np.sqrt(ds.GYRO_F3DB_HZ)))[sel]
    ka = ds.ACC_RMS_MG / (ds.ACC_ND_MG * np.sqrt(ds.ACC_F3DB_HZ))
    fig.suptitle("12  B1 convention check from the datasheet itself (ODR ≤ 1.6 kHz): rms/(density·√f₃dB) = "
                 f"{k.min():.2f}–{k.max():.2f} gyro perf., {kn.min():.2f}–{kn.max():.2f} gyro normal, "
                 f"{ka.min():.2f}–{ka.max():.2f} accel\n→ densities are ONE-SIDED, so Allan N = density/√2.  "
                 "Above 1.6 kHz f₃dB stops scaling with ODR and the gyro rms rises (aliasing/other filter).",
                 weight="bold")
    fig.tight_layout()
    return _save(fig, "12_b1_noise_vs_odr.png")


def fig13_b2_turn_on_bias():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    _watermark(fig)
    cases = [
        (axs[0], "gyro", ds.GYRO_ZERO_RATE_OFFSET_DPS, ds.GYRO_TCO_DPS_PER_K, "dps",
         _nominal("gyroscope_noise_density") / ds.DEG / np.sqrt(DT)),
        (axs[1], "accel", ds.ACC_ZERO_G_OFFSET_MG, ds.ACC_TCO_MG_PER_K, "mg",
         _nominal("accelerometer_noise_density") / (1e-3 * ds.G0) / np.sqrt(DT)),
    ]
    for ax, name, bound, tco, unit, white_sd in cases:
        x = np.linspace(-1.6 * bound, 1.6 * bound, 600)
        ax.axvspan(-bound - 10 * tco, bound + 10 * tco, color=C_SPEC, alpha=0.08,
                   label=f"+ TCO × 10 K (±{tco} {unit}/K)")
        ax.axvspan(-bound, bound, color=C_SPEC, alpha=0.18,
                   label=f"datasheet ±{bound} {unit} (Typ column; σ not stated)")
        for s, ls, lab in [(bound / 3, "-", "if ±bound = 3σ"), (bound, "--", "if ±bound = 1σ")]:
            pdf = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
            ax.plot(x, pdf / pdf.max(), color=C_SPEC, ls=ls, lw=1.2, label=lab)
        ax.axvline(0, color=C_MODEL, lw=3, label="current model: exactly 0 (initial_bias)")
        ax.axvspan(-white_sd, white_sd, ymin=0, ymax=0.08, color=C_MODEL, alpha=0.5,
                   label=f"±1σ per-sample white noise at 200 Hz ({white_sd:.3g} {unit})")
        ax.set_xlabel(f"turn-on bias [{unit}]"), ax.set_yticks([])
        ax.set_title(f"{name}: turn-on bias distribution")
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("13  B2 turn-on bias: datasheet allows ±0.5 dps / ±20 mg; the model has none.\n"
                 "Accept (once D2 exists): model σ_b0 inside the 95% χ² CI of measured σ (≈±20% at 87 dof)",
                 weight="bold", color=C_FAIL)
    fig.tight_layout()
    return _save(fig, "13_b2_turn_on_bias.png")


def fig14_b3_scale_factor():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _watermark(fig)
    bound = params.SCALE_FACTOR_BOUND
    rows = [
        ("gyro sensitivity error (no CRT)", ds.GYRO_SENS_ERR, C_SPEC, bound),
        ("gyro sensitivity error (after CRT)", ds.GYRO_SENS_ERR_CRT, C_SPEC, bound),
        ("gyro TCS × 10 K", ds.GYRO_TCS_PER_K * 10, C_SPEC, None),
        ("gyro cross-axis", ds.GYRO_CROSS_AXIS, C_SPEC, 0.0),
        ("accel sensitivity error", ds.ACC_SENS_ERR, C_SPEC, bound),
        ("accel TCS × 10 K", ds.ACC_TCS_PER_K * 10, C_SPEC, None),
        ("accel cross-axis", ds.ACC_CROSS_AXIS, C_SPEC, 0.0),
    ]
    for i, (lab, spec, col, model) in enumerate(rows):
        y = len(rows) - i
        ax.plot([-spec * 100, spec * 100], [y, y], color=col, lw=7, alpha=0.6, solid_capstyle="butt")
        if model is not None:
            if model > 0:
                ax.plot([-model * 100, model * 100], [y - 0.28] * 2, color=C_MODEL, lw=2.5)
                verdict = model <= 3 * spec
                ax.text(model * 100 + 0.1, y - 0.28,
                        f"model box ±{model:.0%} = {model / spec:.1f}× datasheet "
                        f"→ {'OK' if verdict else 'violates 3× rule'}",
                        va="center", fontsize=7, color=C_TRUTH if verdict else C_FAIL)
            else:
                ax.plot([0], [y - 0.28], "x", color=C_FAIL, ms=8)
                ax.text(0.1, y - 0.28, "not in model", va="center", fontsize=7, color=C_FAIL)
    ax.set_yticks(range(len(rows), 0, -1), [r[0] for r in rows])
    ax.axvline(0, color="0.3", lw=0.8)
    ax.plot([], [], color=C_SPEC, lw=7, alpha=0.6, label=f"datasheet ± ({ds.DOC})")
    ax.plot([], [], color=C_MODEL, lw=2.5, label="model SCALE_FACTOR_BOUND")
    ax.set_xlabel("fractional error [%]")
    ax.set_xlim(-2.3, 4.6)
    ax.legend(loc="lower right")
    ax.set_title("14  B3 scale factor / cross-axis: model search box vs datasheet\n"
                 "Accept (once D3/D4 exist): all |S_meas| ≤ bound ≤ 3·max|S_meas|; cross-axis ≤ spec\n"
                 "(3× rule shown here against the datasheet as a stand-in for max|S_meas|)",
                 weight="bold")
    return _save(fig, "14_b3_scale_factor.png")


def fig15_b4_episode_drift():
    N, K = _nominal("gyroscope_noise_density"), _nominal("gyroscope_random_walk")
    n = int(LOG_HOURS * 3600 / DT)
    x = _trace(77, n, NoiseCoeffs(N, 0.0, K))
    w, a = int(T_EP / DT), int(1.0 / DT)
    nw = n // w
    seg = x[: nw * w].reshape(nw, w)
    db = seg[:, -a:].mean(axis=1) - seg[:, :a].mean(axis=1)
    rms = np.sqrt(np.mean(db**2))
    pred = np.sqrt(2 * N**2 / 1.0 + K**2 * (T_EP - 1.0))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _watermark(fig)
    ax.hist(db / ds.DEG * 1e3, bins=40, color="0.75", edgecolor="white", density=True,
            label=f"model sim: {nw} windows of T_ep={T_EP:.0f} s from an {LOG_HOURS:.0f} h trace")
    for r, ls, lab in [(rms * 0.67, "--", "B4 accept: RMS_real / RMS_sim ∈ [0.67, 1.5]"), (rms * 1.5, "--", None)]:
        for sgn in (-1, 1):
            ax.axvline(sgn * r / ds.DEG * 1e3, color=C_PASS, lw=1.2, ls=ls, label=lab if sgn > 0 else None)
    for dT, ls in [(2, ":"), (5, "-.")]:
        v = ds.GYRO_TCO_DPS_PER_K * dT * 1e3
        ax.axvline(v, color=C_SPEC, lw=1.2, ls=ls, label=f"datasheet TCO × {dT} K warm-up = {v:.0f} mdps")
    ax.set_xlabel("Δb over one episode [mdps]   (mean of last 1 s − first 1 s)")
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"15  B4 episode-horizon gyro drift: model RMS {rms / ds.DEG * 1e3:.1f} mdps "
                 f"(analytic {pred / ds.DEG * 1e3:.1f}; white-noise dominated)\n"
                 "A 2–5 K warm-up allowed by the datasheet TCO is 3–8× larger → cold-start episodes "
                 "are the risk", weight="bold")
    ax.set_xlim(-90, 90)
    return _save(fig, "15_b4_episode_drift.png")


def fig16_b5_delay():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    _watermark(fig)
    step = 1e3 / F_POL
    for ax, odr, gd, f3, name in [
        (a, ds.GYRO_ODR_HZ, ds.GYRO_GROUP_DELAY_MS, ds.GYRO_F3DB_HZ, "gyro (Table 13)"),
        (b, ds.ACC_ODR_HZ, ds.ACC_GROUP_DELAY_MS, ds.ACC_F3DB_HZ, "accel (Table 9)"),
    ]:
        ax.axhspan(0, step, color=C_BAND, alpha=0.7,
                   label=f"B5 accept: |delay_model − delay_meas| ≤ 1 policy step ({step:.0f} ms)")
        ax.semilogx(odr, gd, "o-", color=C_SPEC, label=f"datasheet group delay, {name}")
        ax.axhline(0, color=C_MODEL, lw=2.5, label="model: 0 ms (no filter, no latency)")
        i = np.argmin(np.abs(odr - 1 / DT))
        ax.annotate(f"sim ODR {odr[i]:g} Hz: {gd[i]} ms = {gd[i] / step:.0%} of a policy step\n"
                    f"f₃dB = {f3[i]} Hz = {f3[i] * DT:.2f}·ODR (model: no roll-off)",
                    xy=(odr[i], gd[i]), xytext=(0.35, 0.6), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->"), fontsize=7.5)
        ax.set_xlabel("ODR [Hz]"), ax.set_ylabel("group delay [ms]")
        ax.legend(fontsize=6.8, loc="upper right")
    fig.suptitle("16  B5 signal-chain delay: datasheet vs model (sensor only; MCU/bus/host latency adds on top)",
                 weight="bold")
    fig.tight_layout()
    return _save(fig, "16_b5_group_delay.png")


def fig17_error_budget():
    Ng, Kg = _nominal("gyroscope_noise_density"), _nominal("gyroscope_random_walk")
    Na, Ka = _nominal("accelerometer_noise_density"), _nominal("accelerometer_random_walk")
    tp = 1.0 / F_POL
    gyro = [
        ("white, per sample (model, 200 Hz)", Ng / np.sqrt(DT), True),
        ("white, per sample (datasheet Table 12, 200 Hz)", ds.GYRO_RMS_PERF_MDPS[3] * 1e-3 * ds.DEG, False),
        (f"white, averaged over 1 policy step ({tp * 1e3:.0f} ms)", Ng / np.sqrt(tp), True),
        (f"random-walk drift over T_ep={T_EP:.0f} s", Kg * np.sqrt(T_EP), True),
        ("turn-on bias, datasheet ±bound", ds.GYRO_ZERO_RATE_OFFSET_DPS * ds.DEG, False),
        ("turn-on bias, if bound = 3σ (1σ)", ds.GYRO_ZERO_RATE_OFFSET_DPS / 3 * ds.DEG, False),
        ("thermal offset, TCO × 10 K", ds.GYRO_TCO_DPS_PER_K * 10 * ds.DEG, False),
        (f"scale factor 2 % (no CRT) × {BODY_RATE} rad/s", 0.02 * BODY_RATE, True),
        (f"scale factor 0.4 % (CRT) × {BODY_RATE} rad/s", 0.004 * BODY_RATE, True),
        (f"cross-axis 0.2 % × {BODY_RATE} rad/s", ds.GYRO_CROSS_AXIS * BODY_RATE, False),
        ("g-sensitivity 0.1 dps/g × 1 g (max)", ds.GYRO_G_SENS_DPS_PER_G * ds.DEG, False),
        ("quantization LSB/√12 (±2000 dps)", ds.GYRO_LSB_DPS_2000 * ds.DEG / np.sqrt(12), False),
    ]
    mg = 1e-3 * ds.G0
    accel = [
        ("white, per sample (model, 200 Hz)", Na / np.sqrt(DT), True),
        ("white, per sample (datasheet Table 8, 200 Hz)", ds.ACC_RMS_MG[4] * mg, False),
        (f"white, averaged over 1 policy step", Na / np.sqrt(tp), True),
        (f"random-walk drift over T_ep", Ka * np.sqrt(T_EP), True),
        ("zero-g offset, datasheet ±bound", ds.ACC_ZERO_G_OFFSET_MG * mg, False),
        ("thermal offset, TCO × 10 K", ds.ACC_TCO_MG_PER_K * 10 * mg, False),
        ("sensitivity 0.4 % × 1 g", ds.ACC_SENS_ERR * ds.G0, True),
        ("cross-axis 1 % × 1 g", ds.ACC_CROSS_AXIS * ds.G0, False),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5))
    _watermark(fig)
    for ax, rows, unit, title in [(a, gyro, "rad/s", "gyro"), (b, accel, "m/s²", "accel")]:
        y = np.arange(len(rows))[::-1]
        vals = np.array([r[1] for r in rows])
        cols = [C_MODEL if r[2] else C_FAIL for r in rows]
        ax.barh(y, vals, color=cols, alpha=0.75)
        for yi, v in zip(y, vals):
            ax.text(v * 1.15, yi, f"{v:.2g}", va="center", fontsize=7)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlim(vals.min() / 3, vals.max() * 10)
        ax.set_xlabel(f"magnitude at the policy [{unit}]")
        ax.set_title(title)
    from matplotlib.patches import Patch

    a.legend(handles=[Patch(color=C_MODEL, alpha=0.75, label="in the model / search space"),
                      Patch(color=C_FAIL, alpha=0.75, label="NOT in the model (or not what the model uses)")],
             loc="lower right", fontsize=7)
    fig.suptitle(f"17  Error budget at the policy (§A1): T_ep={T_EP:.0f} s, IMU 200 Hz, policy {F_POL:.0f} Hz, "
                 f"body rate {BODY_RATE} rad/s (assumed)\nThe largest per-episode terms are the ones the model "
                 "lacks; the random walk the Allan anchor validates is near the bottom", weight="bold")
    fig.tight_layout()
    return _save(fig, "17_error_budget.png")


FIGS = {
    "01": fig01_generator_psd, "02": fig02_estimator_agreement, "03": fig03_recovery,
    "04": fig04_null_flicker, "05": fig05_edf, "06": fig06_coverage, "07": fig07_mutations,
    "10": fig10_b1_gyro, "11": fig11_b1_accel, "12": fig12_noise_vs_odr,
    "13": fig13_b2_turn_on_bias, "14": fig14_b3_scale_factor, "15": fig15_b4_episode_drift,
    "16": fig16_b5_delay, "17": fig17_error_budget,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated figure numbers, e.g. 03,10")
    keys = [k.strip() for k in ap.parse_args().only.split(",") if k.strip()] or list(FIGS)
    for k in keys:
        FIGS[k]()


if __name__ == "__main__":
    main()
