# [TITLE]

<!--
SCAFFOLD. Headings, word budgets, and every measured number you might need,
with its source. Prose is yours; this exists so nothing measured gets lost.

Working titles, pick or replace:
  - "I built a differentiable IMU model to break a robot. It didn't work,
     and the reason was more interesting."
  - "Standard domain randomization over-covers sensor noise by 146x"
  - "The error budget that actually breaks a quadruped policy"

Target ~2,800 words. Numbers: docs/findings.md. Narrative: docs/approach_log.md.
Figures: reports/figures/.
-->

## 1. The question and the short answer
<!-- ~250 words. Verdict in the first paragraph. Do not make the reader earn
it. Open with the table; prose after. -->

| premise | verdict |
|---|---|
| failures exist in the physics-parameterized noise space | **no** — every channel 25–635x clear |
| …but they exist in attitude error | **yes** — onset 4.5°, field stack-up 4.41° |
| gradients flow through the simulator | **no** — correct only to H=10 (0.2 s) |
| randomized smoothing rescues them | **no** — mean indistinguishable from zero |
| adaptive search beats uniform sampling | **yes** — 13x more failures |
| …and finds *worse* failures | **no** — 1.5 % difference |
| …and finds *diverse* failures | **no** — collapsed to one mode |

<!-- State your background once, here, and let it work: you came from sensor
physics, which is why you asked whether the error budget was the right one.
Do not apologise for it being a first robotics project. -->

---

## 2. Why this looked like a good bet
<!-- ~400 words. The motivation, argued FROM sensor physics. -->

<!-- KEY NUMBERS
  Playground gyro DR: uniform +-0.2 rad/s -> std 0.1155 rad/s
  BMI270 gyro white noise @ 50 Hz:        std 7.9e-4 rad/s
  Ratio:                                  146x
  Identical gyro=0.2, gravity=0.05 across SEVEN robots: Go1 joystick,
    Go1 handstand, Go1 getup, G1, Berkeley Humanoid, Apollo, T1 --
    quadrupeds and bipeds alike. A copied default, not a calibration.
  Honest caveat: per-episode IMU bias DR DOES appear in the wider
    literature (RSS 2018 Minitaur). "DR omits structured errors" is true
    of Playground, not of the field.
-->

---

## 3. What I built
<!-- ~500 words. Credibility section. Concrete, not exhaustive. -->

<!-- KEY NUMBERS
  Model: Kalibr conventions, reparameterized so eps is drawn outside and
    gradients flow through distribution parameters.
  Allan self-consistency, 16 realizations x 2e6 samples:
    noise density ratio 1.000 (0.1 % spread)
    random walk ratio 1.01-1.04, 11-12 % spread PER REALIZATION
    -> a single real static log cannot pin random walk better than ~+-15 %
  Independent cross-check: generator and estimator paired against a
    separately written validation package in both directions.
  Mutation tests verified to fail for the right reason:
    sqrt(2) in white density -> N moves to 1.4132, K stays 0.9812
    dropped /3 in RW term    -> K moves to 1.7020, N stays 0.9993
  EuRoC was proposed and rejected: no static segment anywhere (quietest
    10 s window peaks at 0.58 rad/s), 85-145 s records. Forced through, the
    estimator returns 0.025 rad/s/sqrt(Hz) against a published 1.70e-4 --
    150x high. Retained as a figure: allan-euroc-contaminated.png
  Figure: allan-self-consistency.png
-->

---

## 4. What I measured
<!-- ~800 words. Figures carry this; prose connects them. -->

### 4.1 Nothing in the noise space breaks it
<!-- Figures: ofat-all-factors.png, ofat-white-vs-bias.png
  64 episodes x 1000 steps. Baseline return 32.18, fall 0.00, survival 1.00.
  gyro white noise    knee 0.50 rad/s      635x BMI270
  gyro persistent bias no knee in range    57x
  gyro scale error     no knee; 50 % error costs 3 %
  gravity tilt         knee 0.20 (11.3 deg) ~10x
  IMU latency          knee 2 steps (40 ms) ~7x
  The structural hypothesis (persistent bias survives 1/sqrt(N) averaging,
  so should knee earlier) is REFUTED: the two curves sit on top of each
  other and bias is if anything better tolerated.
  Policy trained with noise_level = 0.0 -- no observation DR at all -- so
  the robustness is not DR hardening.
-->

### 4.2 It generalizes to rough terrain
<!-- Figure: ofat-all-factors-rough.png
  Identical class, observation, DR, budget, episode length; only terrain XML
  differs. Rewards comparable (29.62 vs 30.01), so not a weaker policy.
  Every gyro knee unchanged. Only latency moved, 2 steps -> 1.
  Baseline there: fall rate 0.17, survival 0.95.
-->

### 4.3 Corruption-insensitive is not unused
<!-- The most quotable result. Give it the subhead.
  full removal      flat return   flat fall   rough return   rough fall
  gyro              9.86 (-70%)   0.52        10.42 (-67%)   0.41
  gravity           30.25 (-6%)   0.03        10.62 (-66%)   0.92
  both              5.61          0.80        1.90           0.92
  Gyro: essential AND precision-tolerant.
  Gravity: required only where the upright prior fails. Predicted from the
  mechanism BEFORE running the rough case, and confirmed.
  For attitude, wrong information is worse than none: 11 deg of tilt costs
  42 %, removal costs 6 %.
-->

### 4.4 The gradients don't work
<!-- H     AD vs finite diff   sign stability
     10    1.7 %               0.78
     25    78 %                0.78
     50    98 %                0.67
     100   104 %               0.61
  Criterion (C2 < 20 %, C3 > 0.8) fixed in week 1. Correct at 0.2 s,
  meaningless by 0.5 s. NOT exploding -- ||g|| falls as H^-0.39 -- just
  decorrelated. Chaotic-landscape signature (Suh et al., ICML 2022).
  Smoothing over 32 episodes: |mean|/SEM 0.4-2.1, all below 3. No descent
  direction exists.
  Also: Warp backend supports no reverse-mode AD at all; cfg.impl="jax"
  required and slower.
-->

### 4.5 The search that remained
<!-- Figure: search-misalign.png
  240 evaluations per arm, common random numbers, 10 % failure threshold
  fixed in advance, +-5 deg/axis box. Baseline 12.34 (400-step horizon).
  uniform DR  14/240 failures   worst 10.36
  CEM        182/240 failures   worst 10.20
  13x discovery rate, 1.5 % severity gain. CEM mode-collapsed:
  sigma 2.82 -> 0.07 deg by gen 9, |mu| locked at 7.0 deg.
  Box corner 8.66 deg RSS vs an 11.3 deg knee, so nothing in-box costs more
  than 17 % of return. Correct physically; shallow by construction.
-->

---

## 5. Why it failed: Allan variance characterizes integration
<!-- ~400 words. The best idea in the project. One thought, stated plainly. -->

<!-- Averaging time tau IS the independent variable of an Allan curve. Noise
density, bias instability, random walk: all statements about integrating a
rate signal over seconds to hours. Right characterization for inertial
navigation, which is what it was built for.
This policy integrates nothing. Memoryless MLP (512/256/128, SiLU,
tanh-normal), acting instant to instant, re-grounded by contact every step.
The long-tau half of the curve describes a failure mode the task cannot have.
Everything follows: gyro channels flat because nothing accumulates them; the
channels that bind are an instantaneous attitude reference with no
redundancy, and a control-loop stability property. Neither is a noise
parameter.
Note it is NOT sensor fusion: no filter, no history, and gravity is ground
truth in sim rather than estimator output. -->

---

## 6. The error budget that actually binds
<!-- ~450 words. YOUR section. The analysis an ML researcher would not do. -->

<!-- Gravity tilt is a DIRECTION error -- the vector is renormalized, so
common-mode magnitude errors vanish entirely. A uniform scale error produces
exactly zero tilt. Causes, all per-axis:
  temperature: TCO 0.015 dps/K, 0.25 mg/K; TCS 0.02 %/K
  package and PCB stress (why specs read "soldered, lifetime")
  mechanical shock, aging, CRT drifted or never run
  g-sensitivity 0.1 dps/g -- locked to the gait cycle, not noise
  chassis deformation, re-assembly
  IMU mounting misalignment

  tilt source                      degrees
  accel zero-g offset (+-20 mg)    1.15
  accel TCO over 40 K              0.57
  accel cross-axis (1 %)           0.57
  RSS, datasheet only              1.41
  + 3 deg mechanical mounting      4.41
  MEASURED degradation onset       4.54

Gyro comparison is the opposite: in-field drift 0.6-1.0 dps against a knee
at 29 dps.
So: gyro drift is irrelevant in the field; attitude drift is marginal in the
field. And every dominant tilt contributor is persistent, per-axis and
slowly varying -- exactly the structure uniform i.i.d. DR omits. -->

---

## 7. What I'd do next
<!-- ~300 words. A program, not loose ends.
  1. The same experiment on a system that INTEGRATES -- inertial odometry or
     VIO. Where Allan parameters are the right characterization and the
     thesis should hold. The real successor.
  2. Misalignment as a sim-to-real study: measure the actual mounting-error
     distribution on hardware, ask whether policies should train with
     misalignment DR. Nobody randomizes it.
  3. What should DR actually randomize? Retrain with less noise, added
     misalignment and latency; compare transfer. The engineering payoff.
  4. Recurrent vs feedforward. Two pre-registered predictions: memory should
     make bias WORSE (a constant offset accumulates) and latency BETTER.
     Cheapest; note brax PPO has no recurrence, so the feasible version is
     frame stacking, which tests latency well and bias weakly.
  Close forward. Last line should make a reader want to know what happens on
  a system that integrates. -->

---

## 8. Limitations
<!-- ~300 words. Straight from docs/findings.md. Do not soften.
  - Severity ceiling: nothing in the defensible box costs >17 % of return.
    "Failure" means degradation, not collapse.
  - Single seed throughout; GPU results are not bit-reproducible (one point
    read 0.00 / 0.031 / 0.00 across three identical-seed runs; ~2/64
    episodes of run-to-run variation). The 13x ratio most deserves a seed
    sweep.
  - One architecture. The latency result is architecture-bound.
  - Simulation only. No hardware validation of any policy claim.
  - Raw sensor, not estimator. The channels that bind are exactly the ones
    that would live in a real state estimator.
  - BMI270 parameters rest on a datasheet plus ONE community measurement of
    one device inside a camera. No own-device static log was recorded.
  - Bench noise, not installed noise. A gyro on a trotting quadruped sees
    vibration and impact transients no Allan curve captures.
  - The simulator was chosen for differentiability, a capability the project
    ultimately could not use.
-->

---

## Appendix: prediction record
<!-- Include this. It reads as rigor when framed as a finding.
  Pre-registered OFAT predictions: 2 of 5.
    right: gyro white no failure; latency most sensitive
    wrong: gyro bias "marginal, decides it" (no knee at all)
           gyro scale "possible" (50 % error costs 3 %)
           gravity tilt "no failure" (one of only two that do)
  Plus three consecutive wrong directional calls:
    persistent bias would decide it; gravity tilt would not fail;
    rough terrain would shrink the gyro margins.
  The ONE correct prediction -- gravity ablation would hurt more on rough
  terrain -- came from a mechanism inferred from the measurements, not from
  sensor-physics intuition.
  That asymmetry IS the argument for measuring sensitivity before building
  a search. -->
