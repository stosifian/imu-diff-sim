"""Does the motion profile move the knees? No -- the reward term does.

The question these figures answer, and the answer, which changed. The sweeps
were first run under three pinned commands against the env's own joystick
command (a lazy random walk: resample ~Exp(5 s), per-axis uniform
+-[1.5, 0.8, 1.2] with per-axis zeroing, standing still only 2 % of the
time). Standing still looked far more sensitive than every other regime.

It was not. Go1JoystickFlatTerrain carries a `stand_still` cost -- weight
-1.0, equal to sum|qpos - default_pose| over the 12 joints -- gated on
||command|| < 0.01 (joystick.py:492). That gate is true for 100 % of a
pinned (0,0,0) sweep and ~1.5 % of the sampled one, so the standing curve
was being charged for something no other curve paid. Re-pinned at
(0.02, 0, 0) -- the same standing robot, one hundredth of a m/s above the
gate -- the sensitivity disappears, while velocity-tracking MSE, fall rate
and survival are unchanged at every magnitude. The robot never behaved
differently; only the metric did.

Design notes, so the choices are auditable rather than taste:

  Form. Continuous sweeps of one magnitude against one metric, so a line per
  series. Two factors in columns, two metrics in rows: return and fall rate
  are different scales and never share an axis.

  Normalisation. Pinning the command changes the baseline -- standing gated
  earns 32.95, ungated 39.22, walking at 1.5 m/s 28.95 -- so every curve is
  a fraction of its OWN baseline. A raw-return comparison across regimes
  would be meaningless, and the gated/ungated pair would be incomparable.

  Colour, and why there is no fifth hue. Hue carries the motion regime
  (slots 1-3 of the validated palette, fixed order); the unpinned sweep is
  the reference the project already reported, so it is recessive grey rather
  than a peer. The two standing runs are the SAME regime measured two ways,
  so they share the regime's hue and are separated by line style -- a
  composite encoding, not a cycled palette. Every series is directly
  labelled as well as in the legend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_ofat import (  # noqa: E402
    AQUA, BLUE, GRID, INK, INK_2, INK_MUTED, ORANGE, SURFACE,
    BMI270_BIAS, BMI270_TILT, BMI270_WHITE, DR_WHITE, _style, _vref, wilson,
)

FACTORS = (
    ("gyro_bias", "gyro persistent bias", "rad/s", BMI270_BIAS,
     "BMI270 turn-on bias bound"),
    ("gravity_tilt", "gravity tilt", "unit-vector offset", BMI270_TILT,
     "BMI270 zero-g offset tilt"),
)


def _declutter(ax, ends, x) -> None:
    """Direct labels, pushed apart so four curves ending together stay legible.

    Identity is never by colour alone here -- the aqua slot carries a sub-3:1
    contrast WARN -- so the labels have to survive the curves converging.
    """
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.052 * (hi - lo)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, name, colour in ends:
        ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points",
                    color=colour, fontsize=8, va="center",
                    annotation_clip=False)


def _load(path: Path) -> dict:
    d = dict(np.load(path))
    d["_n"] = int(d["episodes"]) if "episodes" in d else 64
    return d


def figure(runs, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex="col",
                             constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 0.88, 0.90))
    fig.patch.set_facecolor(SURFACE)

    for col, (factor, label, unit, ref, ref_label) in enumerate(FACTORS):
        for row in (0, 1):
            _style(axes[row, col])

        ends = []
        for name, colour, ls, d in runs:
            if f"{factor}__magnitudes" not in d:
                continue
            is_ref = colour == INK_MUTED
            n = d["_n"]
            m = d[f"{factor}__magnitudes"]
            base = float(d["baseline__return"])
            r = d[f"{factor}__return"] / base
            sem = d[f"{factor}__return_sd"] / np.sqrt(n) / base
            ax = axes[0, col]
            ax.fill_between(m, r - sem, r + sem, color=colour, alpha=0.16,
                            lw=0, zorder=2)
            # The reported sweep is the reference the others are read
            # against, so it is dashed and drawn on top: otherwise a regime
            # that lands on it hides it completely.
            ax.plot(m, r, color=colour, lw=2.2 if is_ref else 2.0, ls=ls,
                    marker="o", ms=5, mec=SURFACE, mew=1.5,
                    zorder=6 if is_ref else 4, label=name)
            ends.append([float(r[-1]), name, colour])

            f = d[f"{factor}__fell"]  # noqa: E741
            lo, hi = wilson(f, n)
            ax = axes[1, col]
            ax.fill_between(m, lo, hi, color=colour, alpha=0.16, lw=0, zorder=2)
            ax.plot(m, f, color=colour, lw=2.0, ls=ls, marker="o", ms=5,
                    mec=SURFACE, mew=1.5, zorder=4, label=name)

        _declutter(axes[0, col], ends, m[-1])

        for row, ylab in ((0, "return, fraction of own baseline\nband: \u00b11 SEM"),
                          (1, "fall rate\nband: Wilson 95%")):
            ax = axes[row, col]
            _vref(ax, ref, ref_label)
            ax.set_xscale("log")
            if col == 0:
                ax.set_ylabel(ylab, color=INK_2, fontsize=9)
        axes[0, col].axhline(1.0, color=GRID, lw=1.0, zorder=1)
        axes[0, col].set_title(label, color=INK, fontsize=10, pad=6)
        axes[1, col].set_ylim(-0.03, 1.03)
        axes[1, col].set_xlabel(f"magnitude [{unit}]", color=INK_2, fontsize=9)

    # Legend in the fall-rate panel: every series is directly labelled in the
    # return row, and there it collided with the BMI270 reference label.
    leg = axes[1, 0].legend(fontsize=8, labelcolor=INK_2, loc="upper left",
                            handlelength=3.0, framealpha=1.0,
                            facecolor=SURFACE, edgecolor="none")
    leg.set_zorder(8)
    # One objective title line, no interpretive subtitle: the reading is
    # argued in the write-up rather than asserted on the figure.
    fig.text(0.01, 0.975,
             "Return and fall rate against injected error magnitude, by "
             "pinned joystick command and by\nwhether the task's "
             "stand_still cost is active: flat terrain, 64 episodes "
             "\u00d7 1000 steps",
             color=INK, fontsize=11.5, va="top", linespacing=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


def structural(gated, ungated, out: Path) -> None:
    """Where the 1/sqrt(N) ordering reversed, and why it was not the regime.

    Section 4.1 refutes the structural hypothesis that persistent bias should
    break the policy before i.i.d. white noise, because only bias survives
    averaging over an episode. Under a pinned standing command the ordering
    appeared to reverse -- bias knee'd first, at 0.092 against white's 0.168
    rad/s -- which made 4.1's refutation look conditional on the robot moving.

    It was conditional on the metric. The reversal is entirely the task's
    `stand_still` cost, which is active only when ||command|| < 0.01.
    Re-pinned one hundredth of a m/s above that gate, white knees first
    (0.251 against 0.418 rad/s) exactly as it does under the sampled command.

    Hue carries the error mode and line style carries whether the
    `stand_still` cost is in the metric, so the figure needs two categorical
    slots, not four.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 0.82, 0.875))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    ends = []
    conditions = ((gated, "stand_still cost active", (0, (5, 2, 1, 2)), 0.10),
                  (ungated, "stand_still cost inactive", "-", 0.18))
    for d, regime, ls, alpha in conditions:
        base = float(d["baseline__return"])
        n = d["_n"]
        for factor, colour, label in (("gyro_white", BLUE, "white noise"),
                                      ("gyro_bias", ORANGE, "persistent bias")):
            m = d[f"{factor}__magnitudes"]
            r = d[f"{factor}__return"] / base
            sem = d[f"{factor}__return_sd"] / np.sqrt(n) / base
            ax.fill_between(m, r - sem, r + sem, color=colour, alpha=alpha,
                            lw=0, zorder=2)
            ax.plot(m, r, color=colour, lw=2.0, ls=ls, marker="o", ms=5,
                    mec=SURFACE, mew=1.5, zorder=4,
                    label=f"{label}, {regime}")
            ends.append([float(r[-1]), f"{label}, {regime}", colour])
    _declutter(ax, ends, m[-1])

    # Reference labels lifted off the baseline: at the default y they run
    # through the legend, which has to sit lower-left here.
    _vref(ax, BMI270_WHITE, "BMI270 white noise", y=0.30)
    _vref(ax, BMI270_BIAS, "BMI270 turn-on bias bound", y=0.30)
    _vref(ax, DR_WHITE, "Playground DR (std)", y=0.30)
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("injected magnitude [rad/s]", color=INK_2, fontsize=9)
    ax.set_ylabel("return, fraction of own baseline  (band: \u00b11 SEM)",
                  color=INK_2, fontsize=9)
    # handlelength: the stand_still condition is encoded by line style, so the
    # legend swatch has to be long enough to show dash-dot against solid.
    # Opaque surface-coloured frame: the reference lines pass behind the
    # legend here, and a frameless box lets them run through the text.
    leg = ax.legend(fontsize=8, labelcolor=INK_2, loc="lower left",
                    handlelength=3.4, framealpha=1.0, facecolor=SURFACE,
                    edgecolor="none")
    leg.set_zorder(8)
    fig.text(0.01, 0.975,
             "Return against injected gyro error magnitude: white noise and\n"
             "persistent bias at equal units, standing still, with the\n"
             "task's stand_still cost active and inactive (flat terrain, 64 "
             "episodes \u00d7 1000 steps)",
             color=INK, fontsize=11.5, va="top", linespacing=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unpinned", type=Path, default=Path("runs/ofat_go1_clean.npz"))
    ap.add_argument("--stand", type=Path, default=Path("runs/ofat_cmd_stand.npz"))
    ap.add_argument("--yaw", type=Path, default=Path("runs/ofat_cmd_yaw.npz"))
    ap.add_argument("--fast", type=Path, default=Path("runs/ofat_cmd_fast.npz"))
    ap.add_argument("--stand-white", type=Path,
                    default=Path("runs/ofat_cmd_stand_white.npz"))
    ap.add_argument("--stand-ungated", type=Path,
                    default=Path("runs/ofat_cmd_stand_ungated.npz"),
                    help="standing pinned at (0.02, 0, 0), above the "
                         "stand_still gate of ||command|| < 0.01")
    ap.add_argument("--stand-ungated-white", type=Path,
                    default=Path("runs/ofat_cmd_stand_ungated_white.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/figures/ofat-command-regimes.png"))
    args = ap.parse_args()

    # Hue = motion regime; line style = whether the stand_still cost is in the
    # metric. The two standing runs are one regime measured two ways, so
    # they share a hue rather than taking a fifth.
    runs = [
        ("sampled (as reported)", INK_MUTED, "--", _load(args.unpinned)),
        ("standing, stand_still cost active", BLUE, "-", _load(args.stand)),
        ("standing, stand_still cost inactive", BLUE, (0, (1, 1.6)),
         _load(args.stand_ungated)),
        ("yaw 1.0 rad/s", ORANGE, "-", _load(args.yaw)),
        ("forward 1.5 m/s", AQUA, "-", _load(args.fast)),
    ]
    for name, _, _, d in runs:
        cmd = d.get("command")
        base = float(d["baseline__return"])
        print(f"{name:28s} baseline return {base:6.2f}  "
              f"fall {float(d['baseline__fell']):.3f}  command "
              f"{'sampled' if cmd is None or np.isnan(cmd).all() else cmd}")
    figure(runs, args.out)

    missing = [p for p in (args.stand_white, args.stand_ungated,
                           args.stand_ungated_white) if not p.exists()]
    if missing:
        print(f"(missing {missing}; skipping the white-vs-bias figure)")
        return 0
    gated = {**_load(args.stand), **_load(args.stand_white)}
    gated["_n"] = _load(args.stand)["_n"]
    ungated = {**_load(args.stand_ungated), **_load(args.stand_ungated_white)}
    ungated["_n"] = _load(args.stand_ungated)["_n"]
    structural(gated, ungated,
               args.out.with_name("ofat-white-vs-bias-standing.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
