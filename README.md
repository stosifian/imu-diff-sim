# Differentiable IMU Noise Model for Policy Edge-Case Discovery

Testing one claim: **given a physics-parameterized IMU noise model, gradient-informed
search finds a frozen policy's failure modes more efficiently than uniform sampling,
at equal compute budget.**

This is a claim about search efficiency over a physics-parameterized space. It is not
a claim that physics-based noise models train better policies, that the failure modes
found are realized on real hardware, or anything about closing the sim-to-real gap.
See `IMU_PROJECT_BRIEF.md` for full scope.

Status: **week 1, in progress.**

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Simulator dependencies (`brax`, `mujoco-mjx`) are a separate extra and are not needed
for the week-1 noise model:

```bash
pip install -e ".[sim]"
```

Development runs on CPU at small scale; real experiments run on a Colab GPU
(`notebooks/colab_runner.ipynb`).

## Layout

| Path | What it is |
|---|---|
| `src/imu_diff_sim/params.py` | Parameter definitions, BMI270 provenance, search-space transforms |
| `src/imu_diff_sim/noise.py` | The differentiable noise model (Kalibr convention) |
| `src/imu_diff_sim/allan.py` | Overlapping Allan deviation + coefficient fitting |
| `tests/` | The specification. Physical invariants, not just smoke tests. |
| `hardware/bmi270_logger/` | Raw static-log capture for Allan characterization |

## Reproduction

Not yet. Week 4 deliverable: a single command from a clean checkout.
