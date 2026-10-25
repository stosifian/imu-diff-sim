"""One definition of a knee, applied to every sweep. The numbers the write-up quotes.

WHY THIS EXISTS. The write-up quoted thresholds that had been read off the
grid by eye, and it used two incompatible conventions without saying so:
section 4.1 called 11.3 deg the gravity-tilt "knee" (42 % of return already
gone) while section 6 called 4.5 deg the "onset" (5 % gone). Same channel,
same sweep. A reader comparing a "10x margin" in one section against
"marginal" in the other cannot reconcile them, and neither number was
reproducible from the npz without knowing which rule produced it.

DEFINITIONS, used everywhere from here on:

  onset  the magnitude at which mean return first falls 5 % below the
         unperturbed baseline. First detectable degradation.
  knee   the magnitude at which it first falls 10 % below. This is the
         threshold pre-registered for the two-arm misalignment search, so
         reusing it keeps one failure criterion across the project.

Both are the FIRST upward crossing, log-interpolated between the two
bracketing sweep points -- linearly for latency, which is integer-valued in
control steps and has no meaningful log interpolation. A factor whose curve
never crosses inside the swept range reports none, with the worst loss
reached, because "no knee" and "a knee just past the last point" are
different claims.

Margins are the knee over that channel's physical reference, so a margin is
"how many times worse than the real device the policy needs before it loses
a tenth of its return".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from imu_diff_sim.sim.go1_obs import (  # noqa: E402
    BMI270_ACCEL_TURN_ON_BIAS_TILT,
    BMI270_GYRO_TURN_ON_BIAS_BOUND,
    BMI270_GYRO_WHITE_STD_50HZ,
)

CTRL_DT = 0.02
#: Gyro group delay at 200 Hz ODR, the only datasheet-side latency term.
BMI270_GROUP_DELAY_S = 5.97e-3
#: Datasheet gyro sensitivity error without CRT (DS000-08 Table 3).
BMI270_SCALE = 0.02

REFERENCE = {
    "gyro_white": (BMI270_GYRO_WHITE_STD_50HZ, "rad/s", "BMI270 white @50Hz"),
    "gyro_bias": (BMI270_GYRO_TURN_ON_BIAS_BOUND, "rad/s", "turn-on bias bound"),
    "gyro_scale": (BMI270_SCALE, "fractional", "sensitivity error +-2%"),
    "gravity_tilt": (BMI270_ACCEL_TURN_ON_BIAS_TILT, "rad", "zero-g offset tilt"),
    "latency": (BMI270_GROUP_DELAY_S / CTRL_DT, "steps", "group delay @200Hz"),
}


def crossing(m: np.ndarray, loss: np.ndarray, thr: float, log: bool = True):
    """First upward crossing of `thr`, interpolated. None if it never crosses."""
    for i in range(1, len(m)):
        if loss[i] >= thr > loss[i - 1]:
            f = (thr - loss[i - 1]) / (loss[i] - loss[i - 1])
            if log:
                return float(np.exp(np.log(m[i - 1]) + f * np.log(m[i] / m[i - 1])))
            return float(m[i - 1] + f * (m[i] - m[i - 1]))
    return None


def table(path: Path, label: str) -> list[tuple]:
    d = dict(np.load(path))
    base = float(d["baseline__return"])
    rows = []
    for factor, (ref, unit, ref_name) in REFERENCE.items():
        if f"{factor}__magnitudes" not in d:
            continue
        m = d[f"{factor}__magnitudes"]
        loss = 1.0 - d[f"{factor}__return"] / base
        log = factor != "latency"
        onset, knee = crossing(m, loss, 0.05, log), crossing(m, loss, 0.10, log)
        rows.append((label, factor, onset, knee, ref, unit, ref_name,
                     float(loss.max()), float(m.max())))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path, default=[
        Path("runs/ofat_go1_clean.npz"),
        Path("runs/ofat_go1_rough_clean.npz"),
        Path("runs/ofat_cmd_stand.npz"),
        Path("runs/ofat_cmd_stand_white.npz"),
        Path("runs/ofat_cmd_yaw.npz"),
        Path("runs/ofat_cmd_fast.npz"),
    ])
    args = ap.parse_args()

    print("DEFINITIONS" + __doc__.split("DEFINITIONS")[1].split("Margins")[0].rstrip())
    print()
    hdr = (f"{'run':<22}{'factor':<14}{'onset 5%':>12}{'knee 10%':>12}"
           f"{'margin':>10}  reference")
    print(hdr)
    print("-" * len(hdr))
    for path in args.runs:
        if not path.exists():
            print(f"{path.name:<22}MISSING")
            continue
        for (label, factor, onset, knee, ref, unit, ref_name,
             worst, mmax) in table(path, path.stem.replace("ofat_", "")):
            def fmt(v):
                if v is None:
                    return "none".rjust(12)
                return f"{v:12.4g}"
            margin = "none" if knee is None else f"{knee / ref:.0f}x"
            note = ref_name
            if knee is None:
                note += f"  (worst {100 * worst:.1f}% at {mmax:.3g} {unit})"
            print(f"{label:<22}{factor:<14}{fmt(onset)}{fmt(knee)}"
                  f"{margin:>10}  {note}")
            if factor == "gravity_tilt":
                deg = [None if v is None else np.degrees(v) for v in (onset, knee)]
                print(f"{'':<36}{fmt(deg[0])}{fmt(deg[1])}{'':>10}  "
                      "(degrees)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
