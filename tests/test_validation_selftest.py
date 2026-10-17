"""Pipeline self-test (docs/validation_protocol.md, section D) as pytest.

Two kinds of test here:

  * the self-test checks themselves -- thresholds live in selftest.py, so the
    CLI and pytest can never disagree about what "pass" means;
  * MUTATION tests that plant the classic convention bugs (a sqrt(2) in the
    white-noise density, a dropped /3 in the random-walk term) and assert the
    checks FAIL. A test suite that has never been seen failing has not been
    shown to test anything.
"""

import ast
from pathlib import Path

import jax
import numpy as np
import pytest

from imu_diff_sim.validation import adev_jax, selftest, synth

VALIDATION_DIR = Path(selftest.__file__).parent


def _all_pass(checks):
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    assert not failed, "\n".join(failed)


# --------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------

def test_validation_package_does_not_import_model_or_its_allan():
    """The second implementation must not borrow from the first."""
    banned = {"imu_diff_sim.noise", "imu_diff_sim.allan"}
    for path in VALIDATION_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                # Resolve relative imports: level 1 is this package, 2 its parent.
                pkg = {0: "", 1: "imu_diff_sim.validation", 2: "imu_diff_sim"}[node.level]
                full = ".".join(p for p in (pkg, node.module or "") if p)
                names = {full} | {f"{full}.{a.name}" for a in node.names}
            else:
                continue
            assert not names & banned, f"{path.name} imports {names & banned}"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 30, 100, 1000, 1e5])
@pytest.mark.parametrize("p", [0.025, 0.975])
def test_chi2_ppf_matches_scipy(k, p):
    from scipy.stats import chi2

    with jax.enable_x64(True):
        approx = float(adev_jax.chi2_ppf(p, float(k)))
    assert approx == pytest.approx(chi2.ppf(p, k), rel=1e-6)


def test_default_ms_unique_and_capped():
    ms = adev_jax.default_ms(100_000)
    assert ms[0] == 1 and ms[-1] == 10_000
    assert np.all(np.diff(ms) > 0)


def test_adev_ci_brackets_point_and_widens_with_tau():
    with jax.enable_x64(True):
        x = np.asarray(synth.white(jax.random.PRNGKey(0), 100_000, 1e-3, selftest.DT))
    taus, adev, _ = adev_jax.overlapping_adev(x, selftest.DT)
    with jax.enable_x64(True):
        edf = np.asarray(adev_jax.edf_oadev(len(x) + 1, np.round(taus / selftest.DT), 0))
    lo, hi = adev_jax.adev_ci(adev, edf)
    assert np.all(lo < adev) and np.all(adev < hi)
    frac = (hi - lo) / adev
    assert frac[-1] > 10 * frac[0]


def test_fit_can_return_exact_zero_for_absent_term():
    """NNLS support enumeration: pure white noise -> K == 0 exactly, not 1e-30."""
    with jax.enable_x64(True):
        x = np.asarray(synth.white(jax.random.PRNGKey(1), 200_000, 1e-3, selftest.DT))
    taus, adev, _ = adev_jax.overlapping_adev(x, selftest.DT)
    c, _ = adev_jax.fit_nbk(taus, adev, selftest.DT, len(x))
    assert c.N == pytest.approx(1e-3, rel=0.02)
    assert c.B == 0.0 or c.K == 0.0


# --------------------------------------------------------------------------
# Self-test checks (fast)
# --------------------------------------------------------------------------

def test_G1_generator_matches_analytic_psd():
    _all_pass(selftest.check_generator_psd())


def test_C2_estimator_matches_allantools_and_brute_force():
    pytest.importorskip("allantools")
    _all_pass(selftest.check_estimator())


def test_C5_edf_calibrated():
    _all_pass(selftest.check_edf())


def test_C2_cross_check_against_main_allan_module():
    """The other implementation, imu_diff_sim.allan, must agree to 1e-9.

    This is the ONE place the two implementations meet, and it lives in the
    test suite rather than the validation package so that the package itself
    stays independent.
    """
    from imu_diff_sim import allan

    with jax.enable_x64(True):
        x = np.asarray(synth.synth_trace(jax.random.PRNGKey(2), 200_000, selftest.DT,
                                         synth.NoiseCoeffs(1e-3, 2e-4, 2e-5)))
    try:
        taus_a, adev_a, _ = allan.overlapping_allan_deviation(x, selftest.DT)
    except NotImplementedError:
        pytest.skip("imu_diff_sim.allan not implemented yet")
    ms = np.round(taus_a / selftest.DT).astype(np.int64)
    taus_j, adev_j, _ = adev_jax.overlapping_adev(x, selftest.DT, ms)
    np.testing.assert_allclose(taus_j, taus_a, rtol=1e-12)
    np.testing.assert_allclose(adev_j, adev_a, rtol=1e-9)


# --------------------------------------------------------------------------
# Self-test checks (slow: 2e6-sample traces, bootstrap)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_C3_recovers_known_coefficients():
    _all_pass(selftest.check_recovery())


@pytest.mark.slow
def test_C4_null_flicker_interval_reaches_zero():
    _all_pass(selftest.check_null_flicker())


@pytest.mark.slow
def test_C6_bootstrap_coverage():
    _all_pass(selftest.check_coverage())


# --------------------------------------------------------------------------
# Mutation tests: plant the bug, the check must catch it
# --------------------------------------------------------------------------

def test_mutation_sqrt2_in_white_density_is_caught(monkeypatch):
    """One-sided vs two-sided density mix-up: generator off by sqrt(2)."""
    real_white = synth.white
    monkeypatch.setattr(selftest, "white",
                        lambda key, n, N, dt: real_white(key, n, N * np.sqrt(2.0), dt))
    results = {c.name: c.passed for c in selftest.check_generator_psd()}
    assert not results["G1 white one-sided PSD = 2N^2"]


@pytest.mark.slow
def test_mutation_dropped_third_in_rw_term_is_caught(monkeypatch):
    """sigma^2 = K^2 tau instead of K^2 tau / 3: K comes back low by sqrt(3)."""
    import jax.numpy as jnp

    def wrong_design(taus):
        return jnp.stack([1.0 / taus, jnp.full_like(taus, adev_jax.FLICKER_FACTOR_SQ), taus],
                         axis=1)

    monkeypatch.setattr(adev_jax, "_design", wrong_design)
    adev_jax._compiled.cache_clear()
    try:
        results = {c.name: c.passed for c in selftest.check_recovery(n_boot=20)}
    finally:
        adev_jax._compiled.cache_clear()
    assert not results["C3 recover K"]
