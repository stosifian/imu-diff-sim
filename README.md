# Differentiable IMU Noise Model for Policy Edge-Case Discovery

A physics-parameterized, differentiable IMU noise model (Kalibr convention, BMI270
reference part), wired into a frozen MuJoCo Playground Go1 locomotion policy to find
out what sensor error actually breaks it.

**The result: nothing in the component parameter space does.** Gyro white noise needs
348× the datasheet spec to reach a knee; bias and scale factor never knee at all
across the swept range. What does bind is system-level and absent from any datasheet —
mounting misalignment (degradation onset at 4.1–5.1° of attitude error, against an
in-field stack-up of the same size) and sensor latency.

The original premise was that gradient-informed adversarial search would beat uniform
sampling at finding failures. Gradients through the contact-rich simulator turned out
to be unusable at any horizon long enough for a failure to develop: by 1.0 s of
simulated time the autodiff gradient disagrees with a finite difference of the same
function by 98 %. The search arm was built gradient-free instead.

Full method, measurements and limitations: [`reports/writeup.md`](reports/writeup.md).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The noise model, the Allan deviation estimator and their tests need nothing else.
Anything that rolls out a policy also needs the simulator extra:

```bash
pip install -e ".[sim]"
```

Development runs on CPU. The sweeps and the PPO training run on a GPU through Modal:

```bash
modal run modal_app.py::ofat --policy /runs/go1_clean.pkl --episodes 64
modal run modal_app.py::search --policy /runs/go1_clean.pkl --budget 240
```

## Layout

| Path | What it is |
|---|---|
| `src/imu_diff_sim/noise.py` | The differentiable noise model (Kalibr convention) |
| `src/imu_diff_sim/params.py` | Parameter definitions, BMI270 provenance, search-space transforms |
| `src/imu_diff_sim/allan.py` | Overlapping Allan deviation + coefficient fitting |
| `src/imu_diff_sim/sim/` | Go1 observation plumbing and the misalignment model |
| `src/imu_diff_sim/validation/` | Generator/estimator self-tests and the BMI270 spec sheet |
| `scripts/ofat_sensitivity.py` | The one-factor-at-a-time sensitivity sweep |
| `scripts/search_misalign.py` | Uniform sampling against CEM over the misalignment box |
| `scripts/spike_gradients.py` | The gradient-quality check, with its pass criteria in the docstring |
| `modal_app.py` | GPU entry points for training, sweeps and search |
| `runs/*.npz` | The sweep outputs every number in the write-up is read from |
| `tests/` | The specification. Physical invariants, not just smoke tests. |

## Reproduction

Every figure in the write-up regenerates from the committed `.npz` sweeps:

```bash
python scripts/plot_ofat.py
python scripts/plot_ablation.py
python scripts/plot_command_regimes.py
python scripts/plot_search.py
```

Regenerating the sweeps themselves requires a GPU and the trained policy checkpoints,
which are not committed.

## Scope

This is a claim about one frozen policy on one task, measured in simulation. It is not
a claim that physics-based noise models train better policies, that the failure modes
found here are realized on real hardware, or anything about closing the sim-to-real
gap. The model has no own-device hardware validation. Limitations are enumerated in
section 8 of the write-up.
