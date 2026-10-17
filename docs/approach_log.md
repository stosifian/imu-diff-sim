# How we got here

A narrative of approaches taken, what each one showed, and why the project
looks different now than it did in the brief. Written to feed the week-4
write-up, so it records the wrong turns as well as the results -- the wrong
turns are most of what was learned.

Companion documents: `validation_protocol.md` (the independent validation
session's method), `validation_findings_for_model.md` (its findings against
the model), and the dated entries in `notes/`.

---

## 0. The original claim

> Given a physics-parameterized IMU noise model, gradient-informed search
> finds a frozen policy's failure modes more efficiently than uniform
> sampling, at equal compute budget.

A claim about **search efficiency over a physics-parameterized space** -- not
about sensor realism, not about policy robustness, not about sim-to-real. The
policy is trained conventionally and frozen; differentiability is used at
analysis time only.

Planned as four weeks: IMU model, simulator integration, three-arm search
experiment, write-up.

## 1. Constraints that shaped everything

- **No CUDA.** An M1 Pro runs JAX on CPU only. Local development is limited to
  smoke-scale runs; anything real goes to Colab. Later measured: creating the
  Go1 MJX environment on CPU takes **1120 s**.
- **One sensor, one task, one policy.** Held throughout.

## 2. Phase one: the IMU model in isolation

### What was built

Three modules, Kalibr naming so the parameters map onto how the field talks
about IMU noise:

- `params.py` -- parameters with per-field provenance, and the physical/
  unconstrained transforms. Box constraints are **structural**: every
  parameter is tanh-squashed (in log space where it spans decades), so any
  real-valued search vector maps to a valid device and the optimizer never
  needs a clip. Clips are non-smooth and would break the gradients the
  project depends on.
- `noise.py` -- discrete-time Kalibr measurement model, reparameterized so
  the standard normals are drawn outside and passed in. That is what makes
  `loss(params)` an ordinary differentiable function, and it gives common
  random numbers across search arms for free.
- `allan.py` -- overlapping Allan deviation, float64, forced-slope coefficient
  fits, honest confidence intervals.

### What validation actually established

Self-consistency, not validation, and the distinction is load-bearing.
Generating from known parameters and recovering them verifies the
discretisation and the unit conventions. It does not verify the physics.

Over 16 independent realizations at 2e6 samples:

| parameter | recovery | 1-sigma per realization |
|---|---|---|
| noise density | ratio 1.000 | 0.1% |
| random walk | ratio 1.01-1.04 | **11-12%** |

The scatter column is the finding. A single static log -- which is all a real
device gives you -- cannot pin random walk better than about +-15%. An early
5-seed run showed a 10.4% "bias" at ~4 sigma that vanished at 16 seeds; a
significance column was added to the report because of it.

### Two wrong turns here

**EuRoC was proposed as external validation and does not work.** The V1
sequences have no static segment anywhere -- the quietest 10 s window still
peaks at 0.58 rad/s -- and at 85-145 s they are far too short to reach the
bias-instability floor. Allan variance on that data measures the MAV's motion.
Forced through anyway, the estimator returns a gyro noise density of 0.025
rad/s/sqrt(Hz) against the published 1.70e-4: **150x high**. That failure is
retained as a figure, because it calibrates how catastrophic a violated
stationarity assumption is.

**Two of the brief's three "sanity envelope" citations do not exist as
described.** arXiv 2005.08284 is a wheeled-robot calibration paper; arXiv
2603.26740 is a monocular VIO paper. Neither is an Allan characterization.
Suvorkin et al. 2024 is real and is a better fit than the brief suggested
(three MEMS grades, week-long static records). The Luxonis measurement was
verified digit-for-digit.

### Breaking the circularity

`validate_model.py` generates with our code and recovers with our code. A
**consistent** convention error passes that check perfectly while leaving the
model wrong against the datasheet.

An independently written validation package (separate session, cross-checked
against `allantools`) now pairs with ours in both directions, plus mutation
tests that plant the two classic convention bugs and assert the check fails.
Verified they fail for the right reason: a sqrt(2) in the white-noise density
moves N to 1.4132 and leaves K at 0.9812; a dropped /3 moves K to 1.7020 and
leaves N at 0.9993.

## 3. Phase two: the gradient spike

The plan treated **chaotic contact gradients** as the project's main technical
risk, citing Suh et al. (ICML 2022) and the literature on chaotic unrolls. A
one-day spike was scheduled with a pass/fail criterion fixed before running:
finite, within 20% of central differences, sign-stable across 8 noise draws,
and no faster than linear growth of ||g|| in horizon.

### The feared risk did not materialise

On Brax ant across horizons 50-400: gradients finite, analytic-vs-finite-
difference agreement **0.0%**, and `||g|| ~ H^0.51` -- sublinear. Contact
gradients were simply fine.

### The actual blocker

Task loss moves **1.1% at H=100 and 0.1% at H=400 across the entire search
box**. A wider probe found the loss flat until gyro noise density reaches
**1000x nominal**.

The mechanism is not subtle: nominal gyro noise std is 5.0e-4 rad/s against a
true |omega| of 6.1e-2 rad/s, and only 3 of the ant's 27 observation entries
carry IMU data. The other 24 are clean ground-truth state, so the policy
routes around the IMU. Three of six parameters had **exactly zero** gradient,
because the observation contains no linear acceleration at all.

**This inverted the plan's risk assessment.** The threat was never gradient
quality; it was that the task is insensitive to the parameters being searched.
A side benefit: the spike caught a real bug -- `float(jnp.log(...))` in
`_log_bounds` raises `ConcretizationTypeError` under jit, which would have
broken every jitted call site including the whole week-three search.

## 4. Switching to Go1, and what that revealed

Go1 joystick was chosen because its observation is restricted to what a robot
can actually sense: gyro, projected gravity, joint states, last action,
command. No ground-truth pose.

Caveat found on inspection: the 48-dim observation **does** include local
linear velocity, which on real hardware comes from a state estimator. So the
"no ground truth anywhere" framing is weaker than it first appeared.

### The measurement that reframed the project

MuJoCo Playground applies uniform observation noise of **+-0.2 rad/s** to the
gyro. A BMI270 at the 50 Hz control rate has a gyro white-noise std of
**7.9e-4 rad/s**.

| | magnitude |
|---|---|
| Playground gyro DR | std 0.1155 rad/s |
| BMI270 gyro white noise @ 50 Hz | std 7.9e-4 rad/s |
| ratio | **146x** |

And the same `gyro=0.2, gravity=0.05` appears across **seven** robots --
quadrupeds and bipeds alike. These are not per-robot calibrations against a
sensor spec; they are a copied default.

A policy trained this way has been hardened against gyro noise 146x worse than
the real sensor. Searching the physical box for white-noise failures cannot
succeed, for gradient search or uniform sampling alike. The week-three
comparison would have been vacuous, and we know that before building it.

### An overclaim, corrected

The first framing was "standard DR covers white noise and entirely omits the
structured errors real devices exhibit." Checking outside MuJoCo Playground
showed that is **false as a statement about the field**: per-episode IMU bias
randomization is documented in multiple quadruped sim-to-real works, including
the RSS 2018 Minitaur paper. It is accurate about Playground specifically.

The surviving, weaker claim: DR noise models are hand-tuned rather than
physics-derived, and scale factor, temporal correlation, cross-axis coupling
and temperature appear nowhere we have found.

## 5. Turn-on bias: the channel that inverts the ratio

The independent validation session raised seven findings against the model.
The consequential one: `initial_bias()` returned zero, removing the largest
bias term in the model.

Turn-on bias is now a **searched parameter** -- deterministic, signed,
per-axis -- rather than a random draw. That distinction matters: a drawn bias
is nuisance randomness the search must average over, while a searched
parameter is something both the gradient method and uniform DR can target on
equal terms. The search vector went from 6 to 12 dimensions.

| | value |
|---|---|
| gyro turn-on bias bound (datasheet, +-0.5 dps) | 8.73e-3 rad/s |
| gyro random-walk drift over a 20 s episode | 4.74e-5 rad/s |
| ratio | 184x |
| Playground DR **persistent** component (500-step episode) | 5.16e-3 rad/s |
| **turn-on bias vs DR persistent component** | **1.7x** |

That last row is the project's current reason to exist. The 146x
over-coverage applies to white noise; for persistent bias the ratio
**inverts**. Zero-mean i.i.d. noise averages down as 1/sqrt(N) over an
episode, so the entire +-0.2 rad/s DR contributes only ~5e-3 rad/s of
sustained error -- while a constant offset does not average down at all and
integrates into heading and attitude.

**Retracted by section 6.** The 2.4x ratio is arithmetically correct and the
turn-on-bias parameterisation was the right modelling change, but the
inference drawn from it -- that this would be the channel where failures
live -- is wrong. Measured, persistent bias has no knee anywhere in the
tested range, and tracks white noise almost exactly. The ratio describes the
error's magnitude relative to training; it says nothing about whether the
policy is sensitive to it, and this policy is not.

Other findings acted on: noise-density provenance corrected (the datasheet
density is one-sided, no sqrt(2); the Luxonis nominal matches **normal** mode
at 1.23e-4, not performance mode at 8.64e-5), scale-factor bounds split per
sensor (gyro +-2% no-CRT, accel +-0.4% -- a shared 2% made the accel box 5x
the datasheet), and every search-box edge tagged with a source, with the
random-walk box explicitly marked UNSOURCED.

## 6. The OFAT result: the claim does not survive measurement

The gating measurement was an **OFAT sensitivity sweep**: inject idealised
error modes one at a time -- white noise, persistent bias, scale factor,
gravity tilt, latency -- find where each knees, and compare against what a
real BMI270 produces. This inverts the original logic. Instead of modelling
the sensor and then searching for failures, find the failure thresholds first
and ask whether any real device can reach them.

Predictions were recorded before running, so being wrong would be informative.
They were wrong.

| factor | predicted | measured | verdict |
|---|---|---|---|
| gyro white noise | no failure, ~150x margin | knee 0.50 rad/s, **635x** | right |
| gyro persistent bias | **marginal -- decides it** | no knee in range, 57x | **wrong** |
| gyro scale factor | possible | 50% error costs 3% of return | **wrong** |
| gravity tilt | no failure, below DR | knee 0.20, 91% falls at 0.5 | **wrong** |
| latency | likely most sensitive | knee 2 steps, collapse by 3 | right |

Two of five. Setup: 64 episodes x 1000 steps against `/runs/go1_clean.pkl`
(206M steps, reward 30.0). Baseline return 32.18, fall rate 0.00, survival
1.00. Figures: `reports/figures/ofat-white-vs-bias.png`,
`ofat-all-factors.png`.

### The structural hypothesis is refuted

After the 146x finding, the project's remaining reason to exist was the
*structural* argument: domain randomization is zero-mean i.i.d. per step, so
it averages down as 1/sqrt(N) over an episode, while a persistent bias does
not average down at all and integrates into heading. Bias should therefore
break the policy at far lower magnitude than white noise.

It does not. At equal magnitude the two curves sit on top of each other, and
bias is if anything slightly *better* tolerated (0.91 vs 0.82 return at
0.5 rad/s). The white-vs-bias figure was built to display a divergence that
is not there.

### Nothing plausible breaks this policy

Margins between each channel's knee and the BMI270 datasheet value: gyro
white 635x, persistent bias 57x with no knee at all, scale 25x with no knee,
gravity tilt ~10x, latency ~7x. The original claim cannot be tested here
because there are no failures inside the physical box for **any** search
method to find -- gradient-informed or otherwise. The week-three three-arm
comparison would have been vacuous, and that is now measured rather than
argued.

### Two findings that are more interesting than the null

**The robustness is not domain randomization.** This policy was trained with
`noise_config.level = 0.0` -- no observation noise whatsoever -- and still
tolerates 635x the BMI270 gyro noise. So the insensitivity does not come from
DR hardening. It comes from the observation carrying redundant state: the
gyro is 3 of 48 dimensions against 24 of joint encoders, plus contact and the
command, so body motion can be reconstructed from leg kinematics. That is a
stronger statement than the 146x framing.

**The two channels that bind are the ones a datasheet does not describe.**
Gravity tilt is the only absolute attitude reference in the observation --
nothing else tells the robot which way is down, so it has no redundancy to
fall back on. Latency is not an information problem at all; it destabilises
the control loop regardless of signal quality. Both live in the **state
estimator**, not in a noise spec. The raw-sensor-versus-estimator concern
recorded in section 7 as a worry has become a measurement.

So the honest formulation is not "policies are insensitive to IMUs" but:

> IMU *rate* information is redundant on this task; IMU-derived *attitude*
> and *timing* are not. A datasheet-parameterised noise model therefore
> targets the one part of the IMU error budget this policy does not care
> about.

### The gravity channel is mechanical, not a noise parameter

The injection perturbs the **direction** of the projected-gravity unit vector
and nothing else -- the vector is renormalised, so magnitude errors vanish.
"Gravity tilt" therefore names the affected channel, not a physical cause.
Everything that makes the robot's estimate of "down" point wrong produces it:

- per-axis accelerometer zero-g offset
- **differential** per-axis scale error (common-mode does nothing: it
  renormalises away, which is why the earlier caption blaming the +-0.4%
  common-mode scale spec was wrong)
- cross-axis coupling and axis misalignment, which this model does not
  include at all
- **IMU mounting misalignment on the chassis**
- estimator error from any source

So the measured statement is: *the policy tolerates attitude-reference error
to roughly 0.1 rad and breaks past 0.2*, whatever produced it.

**This is the one place a mundane hardware imperfection lands near the
failure threshold**, and it tightens the margin badly:

| gravity offset | angle | return | fall rate |
|---|---|---|---|
| 0.0316 | 1.8 deg | 0.990 | 0.00 |
| 0.0794 | 4.5 deg | 0.949 | 0.00 |
| 0.1995 | 11.3 deg | 0.578 | 0.36 |
| 0.5012 | 26.6 deg | 0.114 | 0.89 |

Benchmarked against the datasheet zero-g offset (+-20 mg = 0.02 rad = 1.1 deg)
the margin is ~10x. But a **3 degree IMU mounting misalignment is 0.052 rad**,
which is within 1.5x of where return begins to drop, and a 5 degree error
(0.0875) sits just below the knee. A few degrees of mechanical mounting error
is ordinary on a hand-built robot, it is a *constant* tilt rather than a
fluctuation, and it is not a datasheet parameter at all.

The implication for the project is uncomfortable and belongs in the write-up:
the whole effort was aimed at Allan-variance noise parameters, and the error
budget that actually threatens this policy is **mechanical and
estimator-side**. Misalignment should be added as its own OFAT factor with a
defensible physical bound before any of this is published.

### How the field sees it

Partially consistent, not settled, and nobody appears to have published this
as a clean result. Latency modelling is described as mandatory for transfer,
which matches. One ablation reports a joint-state noise crossover at
sigma ~ 0.2 rad, the same order as our gyro knee, so the magnitudes are not
anomalous. But a parkour study found removing observation noise broke
obstacle traversal, and a dynamics-randomization case study found transfer
worked with no randomization at all. The evidence is task-dependent.

### The caveat that most threatens this

`Go1JoystickFlatTerrain` is the easiest locomotion task available: flat
ground, no perturbations, no obstacles. Joint-encoder redundancy is strongest
exactly when the terrain is predictable; on rough ground, leg kinematics stop
being a reliable proxy for body attitude. The parkour counterexample is
precisely a harder task where noise did matter.

`Go1JoystickRoughTerrain` is therefore running as a controlled comparison:
identical class, observation, DR, budget and episode length -- only the
terrain XML differs. Prediction, again recorded in advance: the gyro margins
shrink. If all five curves stay as flat as on flat terrain, redundancy was
not the mechanism and the insensitivity needs a different explanation.

### Measurement hygiene learned the hard way

Two defects were caught only by luck, and both are now gates rather than
habits:

- **The first full sweep was invalid.** brax trains this task with
  `normalize_observations=True`, and the sweep built its network with the
  default identity preprocessor, feeding the policy raw observations. It fell
  within 3 s of a 20 s episode at *every* magnitude including zero, and every
  row looked alike. It would have been reported as "only latency matters."
  Caught by reading the zero-perturbation row. The sweep now refuses to run
  if the unperturbed policy cannot stand.
- **The first figures carried no uncertainty.** A fall-rate bump of 2
  episodes in 64 was drawn identically to a real knee; it was noticed only
  because it happened to be non-monotonic. A noise excursion pointing the
  convenient way would have read as a finding. Fall rate now carries Wilson
  95% intervals, return carries +-1 SEM.
- **The sweep is not bit-reproducible on GPU.** With identical seeds, that
  same point has read 0.00, 0.031, 0.00 across three runs: floating-point
  reduction order is not fixed and contact dynamics amplify ~1e-7 differences
  until marginal episodes flip. Run-to-run variation near zero fall rate is
  ~2/64 episodes, on top of sampling error.

## 7. Open questions that could still sink it

*Updated after the OFAT result. Two of these are now answered.*

- **RESOLVED, against us: raw sensor vs state estimator.** This was recorded
  as a worry that a datasheet-parameterised raw-sensor model might target a
  quantity the policy never sees. The sweep confirms it. The two channels
  that bind -- gravity-vector error and latency -- are estimator properties,
  not datasheet noise parameters.
- **RESOLVED: is the task sensitive at all?** No, not within any plausible
  box, and not even for a policy trained without observation noise.


- **Bench noise vs installed noise.** The 146x compares DR against *datasheet*
  noise, measured on a desk. A gyro on a trotting quadruped sees structural
  vibration and impact transients absent from any Allan curve. If installed
  noise is ~0.1 rad/s, the DR level is well calibrated and the framing is
  comparing the wrong two quantities.
- **Raw sensor vs state estimator.** On real hardware, projected gravity comes
  from an estimator fusing IMU with kinematics. What reaches the policy is
  *estimator* error -- correlated and structured, and not the same thing. A
  datasheet-parameterized raw-sensor model may target a quantity the policy
  never sees.
- **No real static log yet.** All BMI270 values rest on a datasheet plus one
  community measurement of one device inside a camera. Hardware (Nano 33 BLE
  Sense Rev2, which carries a BMI270) is in transit.
- **Group delay is unmodelled.** 5.97 ms gyro at 200 Hz, ~30% of a 50 Hz
  policy step. Deferred pending the OFAT latency ranking.
- **Policy quality.** The gradient spike used a randomly initialised MLP. A
  trained gait may be more fragile, though a 1000x insensitivity margin will
  not close on policy quality alone.

## 8. Decisions, with reasons

| decision | reason |
|---|---|
| Colab for all real runs | no CUDA locally; Go1 env creation alone is 1120 s on CPU |
| Deferred sim choice to a spike | biggest unknown, resolved with evidence while slack existed |
| Dropped EuRoC for Allan validation | no static segment, records too short |
| Own-device static log instead | validates against the target device; the brief notes no published BMI270 Allan characterization exists |
| Three search arms, not two | uniform DR alone is the weakest baseline; CEM/ES answers "you only beat random search" |
| Structural box constraints, not clipping | clips are non-smooth and break gradients |
| Turn-on bias searched, not sampled | keeps gradient and DR arms comparable; avoids nuisance randomness |
| Per-sensor scale bounds | a shared 2% made the accel box 5x its datasheet spec |
| Smooth surrogate loss, separate hard failure criterion | task success is a step function with no useful gradient |
| OFAT before any further modelling | answers in an afternoon whether the premise holds at all |
