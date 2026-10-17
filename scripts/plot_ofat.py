"""Figures for the OFAT sensitivity sweep.

Design notes, so the choices are auditable rather than taste:

  Form. Each factor is a continuous sweep of one magnitude against a
  performance metric, so the form is a line per factor. The decisive
  comparison -- white noise against persistent bias at EQUAL magnitude, both
  in rad/s -- gets its own figure rather than being buried in a grid, because
  it is the one plot that decides the project's direction.

  Colour. Categorical slots 1-3 of the validated palette, assigned in fixed
  order, never cycled. Verified with the skill's validator (light surface,
  3 slots): lightness band, chroma floor, CVD separation and normal-vision
  floor all pass; aqua carries a sub-3:1 contrast WARN, which obliges visible
  relief, so every series is directly labelled.

  One axis. Return and fall rate are different scales and never share a y
  axis; they are separate rows.

  Text wears ink, not series colour. Reference lines are recessive grey.

  Light mode only: these are static figures for the repo and the write-up,
  not a themed web page.
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

# Validated categorical palette, fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"

FACTOR_LABEL = {
    "gyro_white": "gyro white noise",
    "gyro_bias": "gyro persistent bias",
    "gyro_scale": "gyro scale error",
    "gravity_tilt": "gravity tilt",
    "latency": "IMU latency",
}
FACTOR_UNIT = {
    "gyro_white": "rad/s",
    "gyro_bias": "rad/s",
    "gyro_scale": "fractional",
    "gravity_tilt": "unit-vector offset",
    "latency": "control steps",
}

# See src/imu_diff_sim/sim/go1_obs.py for provenance.
BMI270_WHITE = 1.116e-4 / (0.02 ** 0.5)          # 7.89e-4 rad/s
BMI270_BIAS = 0.5 * np.pi / 180.0                # 8.73e-3 rad/s, datasheet bound
# Gravity tilt is a DIRECTION error: the perturbed vector is renormalised, so
# a common-mode accelerometer scale error produces exactly zero tilt. The
# datasheet source of a tilt is the zero-g OFFSET (+-20 mg -> ~0.02 rad), or a
# per-axis differential scale error -- not the +-0.4% common-mode spec.
BMI270_TILT = 20e-3                              # rad, from +-20 mg zero-g offset
DR_WHITE = 0.2 / np.sqrt(3.0)                    # 0.1155 rad/s


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)


def _vref(ax, x, label, y=0.04):
    """Recessive reference line. Grey, never a series colour."""
    ax.axvline(x, color=INK_MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.text(x, y, f" {label}", color=INK_MUTED, fontsize=7,
            rotation=90, va="bottom", ha="left",
            transform=ax.get_xaxis_transform(), zorder=5)


def headline(d, out: Path, dr_persistent: float) -> None:
    """White noise vs persistent bias at equal magnitude. The decisive plot."""
    base = float(d["baseline__return"])
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    for factor, color in (("gyro_white", BLUE), ("gyro_bias", ORANGE)):
        m, r = d[f"{factor}__magnitudes"], d[f"{factor}__return"] / base
        ax.plot(m, r, color=color, lw=2.0, marker="o", ms=5,
                mec=SURFACE, mew=1.5, zorder=4,
                label=FACTOR_LABEL[factor])
        # Direct label at the right end: identity never by colour alone, and
        # the relief the contrast WARN obliges.
        ax.annotate(FACTOR_LABEL[factor], (m[-1], r[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")

    _vref(ax, BMI270_WHITE, "BMI270 white noise")
    _vref(ax, dr_persistent, "DR persistent part")
    _vref(ax, BMI270_BIAS, "BMI270 turn-on bias bound")
    _vref(ax, DR_WHITE, "Playground DR (std)")

    ax.axhline(1.0, color=GRID, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("injected magnitude [rad/s]", color=INK_2, fontsize=9)
    ax.set_ylabel("return, fraction of unperturbed", color=INK_2, fontsize=9)
    # Subtitle as the axes title, headline above it via fig.text -- setting
    # both on the axes collides them.
    ax.set_title(
        "Same units, same axis. White noise averages down over an episode;\n"
        "a persistent bias does not -- so the two curves should differ.",
        color=INK_2, fontsize=8.5, loc="left", pad=8)
    fig.text(0.012, 0.975,
             "Does the policy care about noise, or about bias?",
             color=INK, fontsize=12.5, fontweight="bold", va="top")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)  # room for direct labels
    ax.legend(frameon=False, fontsize=8.5, loc="lower left",
              labelcolor=INK_2)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"  {out}")


def grid_figure(d, out: Path) -> None:
    """Small multiples: every factor, return and fall rate. Separate rows --
    different scales never share an axis."""
    factors = [f for f in FACTOR_LABEL if f"{f}__magnitudes" in d]
    base = float(d["baseline__return"])
    n = len(factors)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 7.4),
                             constrained_layout=True)
    # Reserve space top and bottom: constrained_layout does not account for
    # a left-aligned suptitle, and the bottom xlabels were being clipped.
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.895), hspace=0.09)
    fig.patch.set_facecolor(SURFACE)

    for j, factor in enumerate(factors):
        m = d[f"{factor}__magnitudes"]
        logx = factor != "latency"

        ax = axes[0, j]
        _style(ax)
        ax.plot(m, d[f"{factor}__return"] / base, color=BLUE, lw=2.0,
                marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=4)
        ax.axhline(1.0, color=GRID, lw=1.0, zorder=1)
        ax.set_ylim(-0.05, 1.25)
        ax.set_title(FACTOR_LABEL[factor], color=INK, fontsize=10,
                     fontweight="bold", pad=6)
        ax.set_xlabel(f"magnitude [{FACTOR_UNIT[factor]}]",
                      color=INK_2, fontsize=8.5)
        if j == 0:
            ax.set_ylabel("return\n(fraction of unperturbed)",
                          color=INK_2, fontsize=9)

        ax2 = axes[1, j]
        _style(ax2)
        ax2.plot(m, d[f"{factor}__fell"], color=ORANGE, lw=2.0,
                 marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=4)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel(f"magnitude [{FACTOR_UNIT[factor]}]",
                       color=INK_2, fontsize=8.5)
        if j == 0:
            ax2.set_ylabel("fall rate", color=INK_2, fontsize=9)

        for a in (ax, ax2):
            if logx:
                a.set_xscale("log")
            if factor in ("gyro_white", "gyro_bias"):
                _vref(a, BMI270_WHITE if factor == "gyro_white" else BMI270_BIAS,
                      "BMI270")

    fig.text(0.008, 0.990,
             "IMU error sensitivity of a Go1 joystick policy",
             color=INK, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.008, 0.948,
             f"{int(d['episodes'])} episodes x {int(d['horizon'])} steps  |  "
             f"baseline return {base:.1f}, fall rate "
             f"{float(d['baseline__fell']):.2f}, survival "
             f"{float(d['baseline__survived']):.2f}",
             color=INK_2, fontsize=9, va="top")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"  {out}")


def knee_table(d) -> None:
    """Where each factor first costs 10% of the unperturbed return."""
    base = float(d["baseline__return"])
    print(f"\n{'factor':<22}{'10% knee':>14}{'unit':>20}{'reference':>28}")
    print("-" * 84)
    refs = {"gyro_white": ("BMI270 7.89e-4 rad/s", BMI270_WHITE),
            "gyro_bias": ("BMI270 8.73e-3 rad/s", BMI270_BIAS)}
    for f in FACTOR_LABEL:
        if f"{f}__magnitudes" not in d:
            continue
        m, r = d[f"{f}__magnitudes"], d[f"{f}__return"] / base
        hit = np.where(r < 0.9)[0]
        knee = f"{m[hit[0]]:.4g}" if len(hit) else "none in range"
        ref_s = ""
        if f in refs and len(hit):
            name, val = refs[f]
            ref_s = f"{m[hit[0]] / val:.0f}x {name}"
        elif f in refs:
            ref_s = f"> range; {refs[f][0]}"
        print(f"{FACTOR_LABEL[f]:<22}{knee:>14}{FACTOR_UNIT[f]:>20}{ref_s:>28}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    horizon = int(d["horizon"]) if "horizon" in d else 1000
    dr_persistent = DR_WHITE / np.sqrt(horizon)
    print(f"horizon {horizon}; DR persistent component "
          f"{dr_persistent:.3e} rad/s\nwrote:")

    headline(d, args.outdir / "ofat-white-vs-bias.png", dr_persistent)
    grid_figure(d, args.outdir / "ofat-all-factors.png")
    knee_table(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
