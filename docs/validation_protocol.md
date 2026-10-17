# BMI270 noise-model validation protocol

**Status:** thresholds fixed 2026-10-15, before any real data was collected.
Amending a threshold after seeing data requires a dated entry in §H with the
reason. Loosening a threshold until a test passes is not an amendment.

**Consumer of the model:** gradient-informed search for failure modes of a
trained locomotion policy (Brax/MJX). What the model must get right is set by
what reaches that policy within one episode, not by what a navigation-grade
characterization would report.

**Open parameters** (fill in once the sim/task spike lands; every threshold
below is written in terms of them):

| Symbol | Meaning | Placeholder |
|---|---|---|
| `T_ep` | episode length | 20 s |
| `f_imu` | sim IMU rate | 200 Hz |
| `f_pol` | policy rate | 50 Hz |
| obs | what the policy reads: raw gyro/accel, or estimator output | unknown |

**Hardware path is undecided**, so each test is tagged **[BMI]** (needs
BMI270 boards), **[PHONE]** (runs on a phone), or **[EITHER]**.

---

## A. Challenges to the current plan

1. **The Allan "credibility anchor" validates the term the policy barely sees.**
   Over one 20 s episode at 200 Hz with the params.py nominal:

   | Term | Magnitude at the policy |
   |---|---|
   | white noise, per sample | 1.116e-4 · √200 ≈ 1.6e-3 rad/s |
   | random walk, drift over `T_ep` | 1.06e-5 · √20 ≈ 4.7e-5 rad/s |
   | turn-on bias (datasheet zero-rate offset) | ±0.5 dps = 8.7e-3 rad/s (DS000-08 Table 3, printed in the Typ column; σ not stated) |

   Turn-on bias is 60–190× the random-walk drift (bound read as 3σ or 1σ) and is constant across the
   episode, which is exactly what a policy integrates into a tilt error.
   `ImuParams` has no turn-on-bias field and `initial_bias()` returns zero
   or is unimplemented, so **the dominant per-episode error is not in the
   search space.**
   **Resolution:** add signed per-axis `gyroscope_turn_on_bias` /
   `accelerometer_turn_on_bias` to `ImuParams`, bounded by the datasheet
   offset spec through `bound · tanh(u)` (same pattern as scale factor). The
   adversary picks the bias *vector*, because a failure mode is "+0.3 dps on
   pitch", not "large σ". DR samples it uniformly in the same box. Allan
   validation is required only for τ ≤ 10·`T_ep`. The overnight RW fit is a
   secondary result.
   **Cheap confirmation once the sim runs:** one gradient of task loss
   w.r.t. every parameter at nominal. If ∂L/∂K is negligible, freeze K at
   nominal and remove it from the search, and report that as a finding.

2. **A phone is not the device under test.** Phones rarely contain a BMI270.
   Android "uncalibrated" streams still carry factory temperature
   compensation. ODR is capped at about 200 Hz without a special
   permission. Range and LPF can't be set. Timestamps come from the
   HAL/host and arrive in batches. The SoC heats the sensor. You can't power-cycle the IMU, and
   there is exactly one unit. The phone can rehearse the pipeline and run
   B1, B3, B4 and the NIS half of C6. It **cannot** run B2 or B5, or say
   anything about unit-to-unit spread.
   **Better grounding:** ≥3 Arduino Nano 33 BLE Sense Rev2 boards (BMI270
   on board), logging the on-chip FIFO with `sensortime` and the
   temperature register over USB. Add an AS5600 Hall-effect encoder on a
   single-axis rotation stage as the independent motion reference. A
   record turntable is enough for constant rate.

3. **√2 in the datasheet-vs-measurement comparison (CONFIRMED from the
   datasheet, 2026-10-15).** The Kalibr/Allan N equals σ(τ=1 s) and is a
   **two-sided** density. The BMI270 datasheet's density is **one-sided**.
   Its own Tables 10–12 show rms / (density·√f₃dB) = 0.93–1.05 at every
   ODR ≤ 1.6 kHz, for both gyro modes and the accel (figure 12). A
   two-sided reading would need a noise bandwidth below the −3 dB point.
   Consequences:
   - The datasheet-implied gyro N is 0.007 dps/√Hz ÷ √2 = **8.64e-5 rad/s/√Hz**
     in performance mode, and 1.23e-4 in normal mode.
   - The params.py nominal (Luxonis, 1.116e-4) is **1.29× the performance-mode
     value**, outside B1's ±20%. It is **0.91× the normal-mode value**. The
     simplest explanation is that the OAK camera ran the gyro in normal mode.
     "Consistent with the datasheet" is only true once the mode is named.
   - Accel: datasheet-implied N = 0.16 mg/√Hz ÷ √2 = 1.11e-3 m/s²/√Hz;
     the Luxonis value is 1.405e-3 (1.27×).
   - Using f₃dB as the noise bandwidth slightly overstates the density
     (true ENBW ≥ f₃dB), so these datasheet N values are upper estimates.

4. **Validating the nominal says nothing about the edges of the search box.**
   With 3 units, a 90%-coverage normal tolerance interval is about ±5σ wide,
   so measurement can't pin the tails. **Resolution:**
   - Set each edge from a *population* statement: datasheet min/max where
     it exists (offsets, sensitivity error with and without CRT,
     cross-axis). Use the Suvorkin 2024 consumer-MEMS range where the
     datasheet gives only a typical value (noise density) or nothing
     (random walk). Tag each edge DATASHEET / LITERATURE in `params.py`.
   - **Tier every failure found** instead of claiming the whole box is
     physical:

     | Tier | Failure lies inside |
     |---|---|
     | T1 | the envelope of your measured units |
     | T2 | datasheet limits |
     | T3 | the literature range |
     | T4 | none of these (reported, not counted) |

     Headline result: failures in T1–T2 found by gradient search vs DR.
   - If K stays in the search, convert RW-driven failures to their
     equivalent bias (≈ K·√`T_ep`) and judge them against the zero-rate
     offset bound.

5. **The noise.py → allan.py → params round trip is self-consistency, not
   validation.** It shares every convention with the model. The self-test
   (§D) uses an independent generator that is itself checked against
   analytic spectra.

---

## B. Error behaviors the model must get right

| # | Behavior | Why it matters to the policy |
|---|---|---|
| B1 | White noise density, gyro and accel, at the sim ODR and filter mode | Per-sample noise on every observation |
| B2 | Turn-on (run-to-run) bias, gyro and accel | Dominant per-episode error (A1). Constant, so it integrates into tilt/heading error |
| B3 | Scale factor + cross-axis (3×3 sensitivity matrix) | Error proportional to motion, largest in aggressive gaits. Already a search dimension |
| B4 | In-run bias drift over 0–10·`T_ep`, including the warm-up transient | Long or cold-start episodes. The only part of RW/thermal behavior that reaches the policy |
| B5 | Group delay / bandwidth / sample timing | Delay destabilizes closed-loop control. If it's absent from the model, delay failures get blamed on noise parameters |

**Dropped, with justification**

- **Bias instability (flicker floor):** about 1e-5 rad/s, three orders below
  B2, and indistinguishable from RW inside one episode.
- **Long-τ (> 10·`T_ep`) RW characterization:** no path to the policy (A1).
  Kept as a secondary report only.
- **Quantization:** at ±2000 dps the LSB is 1.07e-3 rad/s, so q²/12 adds
  about 4% to white-noise variance, below B1's ±20% threshold. *Gate:*
  re-admit if the sim range is set above ±1000 dps with a lower ODR.
- **Temperature coefficients (TCO/TCS):** untestable without a chamber.
  Their effect is folded into the B2/B3 spread and recorded as a
  limitation.
- **Gyro g-sensitivity, scale-factor nonlinearity, vibration
  rectification:** second-order for a consumer part at these rates.
- **Accel saturation:** *gate:* log peak foot-impact acceleration in the
  sim. Promote to B6 only if it exceeds 0.8× the configured range.

---

## C. Tests

### Data sets

| ID | What | Path |
|---|---|---|
| D1 | static log ≥ 8 h with temperature (fit set) | EITHER |
| D1′ | second static log ≥ 8 h, different day (held-out) | EITHER |
| D2 | ≥ 30 power-ups × ≥ 3 units, 60 s each, ≥ 10 min off between | BMI |
| D3 | turntable at 33⅓ and 45 rpm, each axis ±, 60 s per run. Rate reference = photogate/encoder revolution count over ≥ 5 min (not the turntable's spec) | EITHER |
| D4 | 12-position tumble on a squared jig, 60 s per position | EITHER |
| D5 | encoder sine sweep 0.5–40 Hz, encoder and IMU on the same MCU clock | BMI |
| D6 | ≥ 25 single-axis motion runs of length `T_ep` with encoder truth, plus static segments | BMI (PHONE variant in C6) |

### B1 White noise: IEEE 952 Annex C / IEEE 1554 overlapping ADEV + Welch PSD

- **Method:** fit (N, B, K) on D1 with `adev_jax.fit_with_ci`, using the
  fixed window τ ∈ [2/`f_imu`, 10·`T_ep`]. Evaluate on D1′.
- **Pass, all of:**
  - the model's N lies inside the 95% bootstrap CI of the D1′ fit;
  - |ln(N_model / N_D1′)| ≤ ln 1.2;
  - the D1′ one-sided PSD is flat within ±1 dB over 0.1 Hz – 0.4·`f_imu`,
    or the model reproduces the measured roll-off (→ B5).
- **Independent reference:** the held-out log D1′.
- **Can't detect:** noise that changes under motion or vibration;
  unit-to-unit spread.

### B2 Turn-on bias (repeatability, IEEE 952 §"turn-on to turn-on")

- **Method:** for each D2 power-up, take the mean rate over 10–40 s after
  power-on. Compute the pooled within-unit and between-unit standard
  deviations.
- **Pass:**
  - model σ_b0 lies inside the 95% χ² interval of the measured σ (about
    ±20% at 87 dof);
  - every measured bias lies inside the datasheet zero-rate offset bound
    used as the search-box edge.
- **Independent reference:** a held-out unit (fit on 2 units, test on the
  3rd).
- **Can't detect:** temperature dependence outside lab range; ageing;
  solder-stress shifts after mounting on the robot.
- **PHONE:** not runnable. Report as untested; the bound rests on the
  datasheet alone.

### B3 Scale factor / cross-axis

- **Method, gyro (D3):** S = (ω̄ − b)/ω_ref − 1 per axis. The bias b comes
  from the ± direction pair.
- **Method, accel (D4):** IEEE 1293-style multi-position least squares for
  the full 3×3 sensitivity matrix + bias. Reference: local normal gravity
  (WGS84 formula + free-air correction, ~1e-4 relative).
- **Pass:**
  - every measured |S| ≤ `SCALE_FACTOR_BOUND`;
  - `SCALE_FACTOR_BOUND` ≤ 3× the largest measured |S| (otherwise the box is
    unphysically wide);
  - off-diagonal terms ≤ the datasheet cross-axis spec. If they exceed it,
    the model needs misalignment terms.
- **Can't detect:** rate nonlinearity between the two test speeds;
  temperature coefficient of sensitivity. A single unit gives only a lower
  bound on the box edge.

### B4 Episode-horizon drift

- **Method:** from D1′, form windows of length `T_ep` and compute
  Δb = mean(last 1 s) − mean(first 1 s). Simulate the same statistic from
  the model.
- **Pass:**
  - RMS(Δb_real) / RMS(Δb_sim) ∈ [0.67, 1.5] at steady state;
  - for τ ∈ [1/`f_imu`, 10·`T_ep`], the measured ADEV lies inside the
    model's 95% band at ≥ 90% of τ points.
- **Warm-up:** analyze the first 15 min after power-on separately. If the
  real RMS Δb is more than 2× the model's, the model is declared **invalid
  for cold-start episodes**, and the write-up says so.
- **Can't detect:** thermal transients larger than the lab produces, such
  as motor heat on a real robot.

### B5 Delay / bandwidth

- **Method:** cross-correlate D5 gyro with encoder rate on the shared MCU
  clock to estimate group delay and |H(f)|.
- **Pass:**
  - |delay_model − delay_meas| ≤ 1/`f_pol`;
  - |H(0.5·`f_imu`)| within ±1 dB of the model.
- **Analytic cross-check:** datasheet group delay at 200 Hz ODR, normal
  filter (bwp = 0x02): gyro **5.97 ms** (Table 13), accel **5.4 ms**
  (Table 9). That is about 30% of a 50 Hz policy step. The model has 0 ms
  and no roll-off (f₃dB = 77 Hz = 0.39·ODR for the gyro). Figure 16.
- **PHONE:** not runnable. Host timestamps carry 10–50 ms of unknown
  latency.

### C6 Task-level consistency: NEES / NIS

- **Estimator:** a small error-state attitude KF (gyro propagation; gravity
  update when ||a| − g| < 0.05 g). Q, R and P0 are taken from the model
  parameters (N, K, σ_b0) and not hand-tuned.
- **Real branch:** D6 IMU data.
- **Sim branch:** the encoder trajectory's true ω and gravity are fed as
  `true_imu` into `noise.corrupt_trace`. Same estimator, M ≥ 25 runs.
- **Pass, all of:**
  1. sim ε̄_NEES inside the χ²(3M)/M 95% bounds at ≥ 95% of steps. A
     failure here means an estimator or model bug, not a reality gap;
  2. real ε̄_NEES inside the same bounds at ≥ 90% of steps;
  3. mean-NEES ratio real/sim ∈ [0.5, 2];
  4. NIS mean ratio real/sim ∈ [0.67, 1.5], and innovation autocorrelation
     inside ±1.96/√n at ≥ 95% of lags 1–20.
- **PHONE variant:** NIS only (needs no truth), on static and hand-held
  data. Turntable runs give NEES at revolution crossings only.
- **Can't detect:**
  - errors the filter absorbs (a bias it estimates away);
  - errors on unobserved axes (yaw under gravity-only updates);
  - if the policy reads raw IMU and not this estimator's output, C6 is a
    proxy for the policy, not a test of it.

---

## D. Pipeline self-test (must pass before any real data is trusted)

Code: `src/imu_diff_sim/validation/` (independent of `noise.py` and
`allan.py`, enforced by a test).
Run: `PYTHONPATH=src python -m imu_diff_sim.validation.selftest [--slow]`
or `pytest tests/test_validation_selftest.py [-m slow]`.

| Check | What | Threshold | Result 2026-10-15 |
|---|---|---|---|
| G1 | generator one-sided Welch PSD: white = 2N², flicker = B²/(πf), slope −1 | ±5% / ±10% / ±0.05 | 0.999 / 0.992 / −0.998 ✅ |
| C2 | ADEV vs `allantools.oadev`, vs brute-force loop, vs `imu_diff_sim.allan` | 1e-9 rel | 2e-15 / 4e-15 / pass ✅ |
| C3 | recovery from 2e6 samples (N=1.116e-4, B=1e-4, K=1.06e-5) | N ±5%, B ±20%, K ±25%, and truth inside 95% CI | +0.0% / +2.7% / +7.4%, all inside CI ✅ |
| C4 | null test: no flicker in → B's 95% CI reaches 0 | lower bound = 0 | [0, 1.75e-5] ✅ |
| C5 | per-point EDF: empirical var(σ̂²)/σ⁴ vs 2/edf, 300 seeds × 5 τ × 3 noise types | ±20% | 0.91–1.18 ✅ |
| C6 | bootstrap 95% CI coverage over 200 seeds | ∈ [0.91, 0.99] | N 0.925, B 0.950, K 0.990 ✅ (K at the conservative edge) |
| mut. | planted √2 (density convention) and dropped /3 (RW) must make G1 / C3 fail | must fail | both caught ✅ |

**What the self-test cannot detect:**
- misspecification of the three-term model itself (a Gauss–Markov bump,
  thermal ramps, sample drops);
- anything about the real sensor.

It shows that the *instrument* reads correctly. It says nothing about the
*specimen*.

**Notes from building it:**
- Wilson–Hilferty χ² quantiles alone were 16% low at k=3 and collapsed at
  k=1 in the lower tail, which sets the upper error bar at long τ. They're
  replaced with a Newton-refined quantile tested to 1e-6.
- K's bootstrap interval over-covers slightly (0.990), so it is
  conservative. N's is slightly narrow (0.925): treat N's interval as
  approximately, not exactly, 95%.

---

## E. Where data collection can undermine each test

| Threat | Effect | Mitigation | Hits |
|---|---|---|---|
| **Duration** | 8 h gives about ±30% per-point bars at τ ≈ 800 s. Enough for 10·`T_ep`, not enough to separate flicker from RW (which §B says you don't need) | Don't fit past τ = T/10 (default cap in `default_ms`) | B1, B4 |
| **Temperature drift** | A slow ramp looks like a +½-slope random walk and inflates K and Δb | Log temperature. Reject windows with \|dT/dt\| > 0.5 °C/h. Phone: screen off, nothing running, not charging | B1, B4, B2 |
| **Host timestamps** | Jitter and batching alias into the white-noise level. Delay becomes unmeasurable | Use the BMI270 `sensortime`. Phone: Δt histogram, drop runs with > 0.1% missing samples | B1, B5, C6 |
| **Single unit** | B2 has no variance estimate. B3 gives only a lower bound on the box edge. Every pass means "this unit", not "the BMI270" | ≥ 3 units. Otherwise state it in every result | B2, B3, A4 |
| **Phone as device** | Different part, OS processing, no power-cycle | Use only for pipeline rehearsal, B1/B3/B4 shape, and NIS | all |
| **Fitting and testing on the same log** | Circular. Any fit passes its own data | D1 fit, D1′ test; fit units ≠ test units | B1, B2, B4 |

---

## F. Code map

| File | Role |
|---|---|
| `validation/synth.py` | independent white/flicker (Kasdin–Walter)/RW/quantization generator; PSD convention table |
| `validation/adev_jax.py` | overlapping ADEV (JAX, float64), EDF (NIST SP 1065 §5.3), χ² intervals, NNLS fit of N/B/K, parametric bootstrap CI |
| `validation/selftest.py` | checks G1–C6 with the thresholds above; CLI |
| `tests/test_validation_selftest.py` | the same checks as pytest, plus mutation tests and the cross-check against `imu_diff_sim.allan` |

## G. Diagnostic figures

Generate: `PYTHONPATH=src python scripts/plot_validation.py [--only 03,10]`
→ `reports/figures/validation/`. Self-test figures show real results with
the pass bands drawn in. Spec/model figures carry a **NO MEASURED DATA YET**
watermark. They show the model's prediction, the datasheet limits
(BST-BMI270-DS000-08, transcribed in `validation/bmi270_spec.py`) and each
test's acceptance region, ready for measurements to be plotted onto.

| Fig | Test | What it shows |
|---|---|---|
| 01 | G1 | generator PSD vs IEEE 952 spectra; the √2 line marks what a convention bug looks like |
| 02 | C2 | three ADEV implementations overlaid; differences ~1e-15 vs the 1e-9 limit |
| 03 | C3 | ADEV + χ² bars, fit and components; bootstrap histograms vs truth and tolerance |
| 04 | C4 | white + RW only: the naive argmin "bias instability" vs the fit's B interval reaching 0 |
| 05 | C5 | per-point EDF calibration, 3 noise types, ±20% band |
| 06 | C6 | 200 bootstrap intervals per coefficient, misses in red. Note that K is fitted as exactly 0 in about 12% of 1000 s records: K is weakly identified at that length |
| 07 | mutations | planted √2 and dropped /3, both caught |
| 10, 11 | B1/B4 | model ADEV + 95% band for one 8 h log, vs datasheet-implied N (both modes), Luxonis, search box, B1 acceptance at τ = 1 s, quantization floor |
| 12 | B1 | datasheet rms vs ODR: proves the one-sided convention; the model as implemented (no LPF) vs band-limited |
| 13 | B2 | datasheet turn-on bias ±0.5 dps / ±20 mg (σ unstated, both readings), + TCO × 10 K; the model's zero |
| 14 | B3 | SCALE_FACTOR_BOUND vs datasheet sensitivity/cross-axis; the accel box is 5× the datasheet |
| 15 | B4 | model Δb over T_ep windows with the B4 acceptance band; a datasheet TCO warm-up of 2–5 K is 3–8× larger |
| 16 | B5 | datasheet group delay vs ODR, the model's 0 ms, acceptance band |
| 17 | §A1 | per-episode error budget at the policy, in-model vs not-in-model |

**Not yet drawable:** C6 NEES/NIS (needs the estimator) and all measured
overlays (need D1–D6).

## H. Amendments

*(none yet)*
