# Glossary and clarifications

Running log of terms and definitions asked about during the project, kept so
they can be refreshed without re-reading the conversation. Appended to
whenever a new one comes up.

---

## Sensor and IMU terms

**Allan deviation / Allan variance**
A way of characterising sensor noise across timescales. Take a long recording
from a motionless sensor, average it over windows of length tau, and measure
how much consecutive window-averages differ. Plot against tau. Short tau:
fast random noise dominates and averaging helps, so the curve slopes down
(−1/2). Long tau: slow drift dominates and averaging hurts, so it turns up
(+1/2). The shape identifies which error sources are present and how large.
Implemented in `src/imu_diff_sim/allan.py`; rendered as
`reports/figures/allan-equation.png`.

**Noise density** (`gyroscope_noise_density`, rad/s/sqrt(Hz))
The white-noise term. Note rad/s/sqrt(Hz) and rad/sqrt(s) are the same units
written differently, so the coefficient equals the Allan deviation at
tau = 1 s.

**Random walk** (`gyroscope_random_walk`, rad/s^2/sqrt(Hz))
How fast the sensor's zero point wanders. Variance accumulates linearly in
elapsed time, so bias std after T seconds is sigma_rw * sqrt(T) regardless of
sampling rate.

**Bias instability**
The flat floor in an Allan curve, from 1/f flicker noise. Our model uses a
pure random walk, which has no flicker term, so there is no true floor --
only the crossover between the two slopes.

**Turn-on bias / zero-rate offset**
A constant offset present from power-on, fixed for the session. BMI270
datasheet: +-0.5 dps gyro, +-20 mg accel. In this project it is a *searched*
parameter rather than a random draw, so both search arms can target it on
equal terms.

**Scale factor error**
Multiplicative sensitivity error: the sensor reads (1 + s) x truth. Entered as
a *fraction*, so 10 % is `0.1` and 2 % is `0.02`. Signed — it goes both ways.

**CRT (Component Re-Trimming)**
A Bosch feature that re-calibrates the gyro's sensitivity on-chip to
compensate for aging and mechanical stress. Matters twice: the datasheet
quotes different sensitivity error with it (+-0.4 %) and without (+-2 %), and
a retrim firing mid-recording steps the scale factor, which corrupts an Allan
measurement.

**TCO / TCS**
Temperature coefficients of Offset and of Sensitivity. BMI270: TCO
0.015 dps/K (gyro), 0.25 mg/K (accel); TCS 0.02 %/K (gyro), 0.004 %/K (accel).
Both act per-axis, which is what turns a temperature change into an attitude
error rather than a harmless common-mode one.

**g-sensitivity**
Gyro output error proportional to linear acceleration, 0.1 dps/g on the
BMI270. Under multi-g foot impacts this is a bias *locked to the gait cycle* —
structurally different from noise.

**Cross-axis sensitivity**
One axis responding to motion on another. BMI270: 0.2 % gyro, 1 % accel.

**Error budget**
The itemised accounting of every error source contributing to a system's
total error, each with a magnitude, combined (usually RSS). The habit that
produced this project's main result: *before refining a term, check the
budget is complete.* The IMU budget here was missing its dominant terms --
misalignment and latency -- while the project refined the Allan terms.

**Gravity tilt (as used in this project)**
A *direction* error in the projected-gravity vector, not a magnitude error.
The vector is renormalised, so a common-mode scale error produces exactly
zero tilt. Causes: per-axis zero-g offset, *differential* per-axis scale
error, cross-axis coupling, IMU mounting misalignment, estimator error.

**Kalibr**
An open-source camera/IMU calibration toolbox from ETH Zurich. Its wiki's IMU
noise-model page became the de facto standard for *naming* IMU noise
parameters (`gyroscope_noise_density`, `gyroscope_random_walk`, ...). We
follow that convention so parameters map onto how the field talks.

**Luxonis**
Company making the OAK depth/AI cameras. A user characterised the BMI270
inside one over 10 hours at 100 Hz and posted four Kalibr-format parameters
in a GitHub issue (luxonis/depthai-core#886). Those are our nominal values.
One device, inside a camera body, and the poster doubted their own gyro
random walk — the weakest numbers in the model.

---

## Model and code

**`to_unconstrained` / `from_unconstrained`**
The map between physical parameters (SI units, bounded) and the unbounded
vector an optimizer walks on. `from_unconstrained` is where physical validity
lives: every real-valued vector maps to a plausible device, so no search
method ever needs a clip. Positive quantities spanning decades use tanh
composed with log; signed bounded ones use tanh alone.

**Reparameterisation (why noise is drawn outside the model)**
Instead of generating noise inside the model (`random.normal(0, sigma)`),
unit-variance noise is drawn once outside and the parameters scale it
(`sigma * eps`). Mathematically identical, but sigma now enters as plain
multiplication so gradients can reach it, and with eps fixed two parameter
settings can be compared against *identical* noise.

**Common random numbers**
The consequence of the above: any two points being compared see the same
episode seeds, so differences are attributable to the parameters rather than
sampling luck. Used across OFAT magnitudes and between the two search arms.
Like testing two sensor designs against one recorded vibration profile
instead of two separate shake-table runs.

**`SEARCH_BOX_FACTORS`**
Multiplicative bounds on each noise parameter, expressed as factors on
nominal. Not simply "tolerances": tolerance is a per-part guarantee at spec
conditions, while the box must cover a deployed device (temperature, aging,
no CRT). It is an assertion that every device inside it could plausibly ship.

---

## Simulation and RL

**Go1 joystick**
The Unitree Go1 quadruped tracking a commanded velocity (forward, lateral,
yaw). The standard locomotion task in MuJoCo Playground. Its 48-dim
observation is linear velocity (3), gyro (3), projected gravity (3), joint
angles (12), joint velocities (12), last action (12), command (3).

**Projected gravity**
The gravity direction expressed in the robot's body frame — how it knows
which way is down. Derived from the accelerometer on real hardware; ground
truth in this simulation.

**H (rollout horizon)**
How many simulation steps run before the loss is computed and differentiated
back through. At 50 Hz control: H=10 is 0.2 s, H=100 is 2 s, H=1000 is a full
20 s episode. Longer H stacks more contact events between parameter and loss,
which is why gradient accuracy degrades with it.

**Domain randomization (DR)**
Randomising simulation parameters during training so the policy tolerates
variation. In MuJoCo Playground the observation noise is drawn fresh *every
step*, zero-mean.

**Per-episode bias DR**
The contrasting variant: draw a bias once at `reset()` and hold it constant
for the whole episode. The difference is temporal structure, not magnitude —
per-step noise averages down as 1/sqrt(N), a per-episode bias does not. Used
in parts of the quadruped sim-to-real literature, not in Playground.

**OFAT (one-factor-at-a-time)**
Vary one input across a range with the others fixed, and measure. Unglamorous
and cheap; it redirected this project after four weeks of search
infrastructure assumed its answer.

**CEM (Cross-Entropy Method)**
A gradient-free optimizer: sample a population from a Gaussian, keep the
worst-performing elites, refit the Gaussian to them, repeat. Adaptive but
needs no gradients, which is why it is the fair baseline — it is what a
gradient method would have had to beat.

**Ablation vs corruption**
Corrupting a channel degrades its value; ablating it removes the channel
entirely. The distinction turned out to be the project's most important: the
gyro is immune to 635x its datasheet noise (corruption) yet costs 70 % of
return when removed (ablation). Insensitive is not unused.

**Fall rate**
Fraction of episodes in which the robot flipped at any point. The env's
termination test is literally `upvector[-1] < 0` — the body's up axis below
horizontal — so it is a genuine flip, not a stumble.

---

## Statistics and plots

**Wilson interval (the shaded band on fall-rate plots)**
A confidence interval for a proportion. Used instead of the normal
approximation because fall rate sits at or near zero over most of each sweep,
where the normal interval has *zero width* and would draw "0 of 64" as
certainty. 0/64 is in fact consistent with a true rate up to 5.6 %.

**SEM band (the shaded band on return plots)**
+-1 standard error of the mean across the 64 episodes at that point.

**RSS (root sum of squares)**
How independent error terms combine: sqrt(a^2 + b^2 + ...). Used for the
tilt stack-up, where the datasheet terms RSS to 1.41 deg.

---

## Tooling

**`python -m pytest -q --no-header -x`**
Run the test suite via the current interpreter (`-m` guarantees the same
environment), quietly, without the platform banner, stopping at the first
failure. Note `-x` does not stop on a *skip*, which is how a silently skipped
test slipped past once.

**Why not Gazebo?**
It models sensor physics, but it has no autodiff (the project's premise
needed gradients), runs roughly real-time single-environment where training
needed 206M steps, and its IMU plugin is Gaussian noise plus a bias random
walk — structurally the same model we wrote. The differentiability
requirement that excluded it turned out to be unusable anyway.
