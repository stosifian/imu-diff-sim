# Findings

What the project set out to test, what was measured, and what was learned.
`approach_log.md` is the narrative of how we got here, including the wrong
turns; this is the distillation.

Every number below is from a run in this repo. Where something is inference
rather than measurement it says so.

---

## The claim, and its verdict

> Given a physics-parameterized IMU noise model, gradient-informed search
> finds a frozen policy's failure modes more efficiently than uniform
> sampling, at equal compute budget.

**Not testable on this system, for two independent reasons — either alone is
sufficient.**

| premise | verdict |
|---|---|
| failures exist in the physics-parameterized noise space | **no** — every channel 25–635x clear |
| …but they exist in attitude error | **yes** — onset 4.5 deg, field stack-up 4.41 deg |
| gradients flow through the simulator | **no** — correct only to H=10 (0.2 s) |
| randomized smoothing rescues them | **no** — mean indistinguishable from zero |
| adaptive search beats uniform sampling | **yes** — 13x more failures |
| …and finds *worse* failures | **no** — 1.5 % difference |
| …and finds *diverse* failures | **no** — collapsed to one mode |

---

## 1. Standard DR over-covers sensor noise by two orders of magnitude

MuJoCo Playground applies uniform gyro noise of ±0.2 rad/s (std 0.1155). A
BMI270 at the 50 Hz control rate has a gyro white-noise std of 7.9e-4 rad/s.
**Ratio 146x.** The identical `gyro=0.2, gravity=0.05` appears across seven
robots — quadrupeds and bipeds — so it is a copied default, not a per-robot
calibration against any sensor spec.

Caveat established by checking outside that codebase: per-episode IMU bias
randomization *does* appear in the wider literature, so "DR omits structured
errors" is true of Playground, not of the field.

## 2. The policy is immune to every datasheet noise channel

64 episodes x 1000 steps, trained policy, baseline return 32.18 / fall rate
0.00 / survival 1.00.

| channel | 10 % knee | margin over BMI270 |
|---|---|---|
| gyro white noise | 0.50 rad/s | **635x** |
| gyro persistent bias | none in range | 57x, no knee |
| gyro scale error | none in range | 50 % error costs 3 % |
| gravity tilt | 0.20 (11.3 deg) | ~10x |
| IMU latency | 2 steps (40 ms) | ~7x |

**The structural hypothesis is refuted.** The argument that persistent bias
should break the policy where i.i.d. noise does not — because only the former
survives 1/sqrt(N) averaging — predicted bias would knee far earlier. At
equal magnitude the two curves sit on top of each other, and bias is if
anything *better* tolerated.

**It is not an artifact of easy terrain.** Rough terrain, identical in every
respect but the terrain XML, moved only the latency knee (2 steps to 1).
Every gyro channel was unchanged.

**It is not domain randomization doing the work.** This policy trained with
`noise_level = 0.0` — no observation noise at all — and still tolerates 635x
the BMI270 gyro noise.

## 3. Corruption-insensitive is not unused

The single most important correction in the project. Ablation — removing the
channel rather than degrading it — reverses the ordering:

| full removal | flat return | flat fall | rough return | rough fall |
|---|---|---|---|---|
| gyro | 9.86 (−70 %) | 0.52 | 10.42 (−67 %) | 0.41 |
| gravity | 30.25 (−6 %) | 0.03 | **10.62 (−66 %)** | **0.92** |

**The gyro is essential and precision-tolerant.** The policy needs to know
roughly how fast the body is rotating; it does not need that number accurate.

**Gravity is required only where the upright prior fails.** Removing it costs
6 % on flat ground, where "assume upright" is nearly correct, and 66 % on
rough ground where it is not. Predicted from the mechanism before running,
and confirmed.

**For attitude, wrong beats absent.** Tilting gravity 11 deg costs 42 % while
removing it costs 6 %. This matters because every in-field drift mechanism
produces a *wrong* value, not a missing one.

## 4. The binding error budget is mechanical, not a datasheet spec

"Gravity tilt" is a direction error — the vector is renormalized, so
common-mode magnitude errors vanish entirely. Causes, all acting per-axis:

temperature (TCO 0.015 dps/K, 0.25 mg/K; TCS 0.02 %/K), package and PCB
stress, mechanical shock, aging, CRT drift, g-sensitivity (0.1 dps/g, locked
to the gait cycle), chassis deformation, re-assembly, and **IMU mounting
misalignment**.

| tilt source | degrees |
|---|---|
| accel zero-g offset (±20 mg) | 1.15 |
| accel TCO over 40 K | 0.57 |
| accel cross-axis (1 %) | 0.57 |
| **RSS, datasheet only** | **1.41** |
| **+ 3 deg mechanical mounting** | **4.41** |
| *measured degradation onset* | *4.54* |

A realistic field stack-up lands **at** the threshold. The gyro comparison is
the opposite: in-field drift of ~0.6–1.0 dps against a knee at 29 dps.

## 5. Gradients through MJX contact are unusable at useful horizons

Against a pass/fail criterion fixed in week 1:

| H | AD vs finite differences | sign stability |
|---|---|---|
| 10 | **1.7 %** | 0.78 |
| 25 | 78 % | 0.78 |
| 50 | 98 % | 0.67 |
| 100 | 104 % | 0.61 |

Correct at 0.2 s, meaningless by 0.5 s. Not exploding — `||g||` falls as
H^−0.39 — just decorrelated from the true derivative, the chaotic-landscape
signature. Randomized smoothing over 32 episodes gives |mean|/SEM of 0.4–2.1,
all below 3: **no descent direction exists.**

Also: the Warp backend supports no reverse-mode AD at all (its JAX FFI call
has no transpose rule); `cfg.impl = "jax"` is required and is slower.

## 6. Adaptive search wins on rate, not on severity or diversity

240 evaluations per arm, common random numbers, 10 % failure threshold fixed
in advance, over a ±5 deg/axis misalignment box.

| | uniform DR | CEM |
|---|---|---|
| failures found | 14 / 240 | **182 / 240** |
| worst return | 10.36 | 10.20 |

13x the discovery rate, 1.5 % the severity gain. CEM collapsed to a single
mode (sigma 2.82 → 0.07 deg). The box corner is 8.66 deg against an 11.3 deg
knee, so the physically defensible bound sits below the catastrophic regime
and nothing in-box costs more than 17 % of return.

---

## Why the mismatch

**Allan variance characterizes integration.** Averaging time is its
independent variable; noise density, bias instability and random walk are all
statements about integrating a rate signal over seconds to hours. That is the
right characterization for inertial navigation.

**This policy integrates nothing.** It is a memoryless MLP acting instant to
instant, re-grounded by contact every step. The long-tau half of the Allan
curve describes a failure mode the task structurally cannot have.

We characterized the sensor for integration and tested it on a system that
does not integrate. Every result follows: the gyro channels are flat because
nothing accumulates them; the channels that bind are an instantaneous
attitude reference with no redundancy, and a control-loop stability property.

**Where the thesis should hold instead** (inference, not measurement):
dead-reckoning and VIO, flight phases and aerial robots, recurrent policies
that accumulate IMU over time, humanoids, and estimator-in-the-loop systems.
The brief's thesis is not refuted — its pairing with this system class is.

---

## Methodological learnings

These cost real time and belong in the write-up.

- **Measure sensitivity before building a search.** Four weeks of
  infrastructure assumed failures existed in the search space. One afternoon
  of OFAT showed they did not. The ordering should have been inverted.
- **Intuition was a poor guide here: 2/5 on pre-registered predictions, plus
  three consecutive wrong directional calls.** The one correct prediction
  came from a mechanism inferred from data, not from sensor physics. That is
  the argument for measuring first, stated as evidence rather than modesty.
- **A plausible result from a broken pipeline is the real hazard.** The first
  full sweep fed the policy unnormalized observations; it fell at every
  magnitude including zero, and would have been reported as "only latency
  matters." Caught by reading the zero-perturbation row. The sweep now
  refuses to run if the unperturbed policy cannot stand.
- **Figures without uncertainty invite false findings.** A 2-of-64 fall-rate
  blip was drawn identically to a real knee, and was noticed only because it
  pointed the wrong way. Wilson intervals and SEM bands now ship by default.
- **GPU results are not bit-reproducible.** With identical seeds, one point
  read 0.00 / 0.031 / 0.00 across three runs — contact dynamics amplify
  ~1e-7 reduction-order differences until marginal episodes flip. ~2/64
  episodes of run-to-run variation on top of sampling error.
- **Guard clauses can silently kill gradients.** `jnp.where` around a
  singularity routes the derivative through the selected branch only, giving
  exactly zero gradient at the guarded point — which was the search's
  starting point. Remove singularities rather than hiding them.
- **Cross-check against an independent implementation.** Generator and
  estimator written by the same author agree even when both are wrong; a
  consistent convention error passes a round-trip test perfectly.

---

## Limitations

- **Severity ceiling.** Nothing inside the physically defensible box costs
  more than 17 % of return. "Failure" here means degradation, not collapse.
- **Single seed throughout.** One policy per terrain, one search seed, with
  known GPU non-reproducibility on top. The 13x discovery ratio is the only
  strongly positive number and most deserves a seed sweep.
- **One architecture.** A memoryless MLP, 512/256/128. The latency result in
  particular is architecture-bound: a recurrent policy could compensate for
  delay, and could plausibly be *more* bias-sensitive, since memory is what
  turns a constant offset into growing error.
- **Simulation only.** No hardware validation of any policy claim.
- **Raw sensor, not estimator.** On real hardware, projected gravity comes
  from a filter fusing IMU with kinematics; here it is ground truth. The
  channels that bind are exactly the ones that would live in that filter.
- **BMI270 parameters rest on a datasheet plus one community measurement**
  of one device inside a camera. No own-device static log was recorded.
- **Bench noise, not installed noise.** Allan parameters describe a part on
  a desk. A gyro on a trotting quadruped sees structural vibration and
  impact transients that no Allan curve captures.
