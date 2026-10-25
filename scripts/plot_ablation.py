"""Two figures the write-up needs and the sweep plots do not cover.

1. Ablation: pinning a channel to a constant versus corrupting it, flat
   against rough. The most quotable result in the project had no figure.
2. The in-field tilt error budget against the measured degradation onset.
   The section a sensor-physics background produces had no figure either.

Design follows the same rules as plot_ofat.py: form chosen before colour,
validated categorical slots in fixed order, no dual axes, direct labels
alongside the legend, recessive grid, light surface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_ofat import wilson  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)


def ablation(flat: Path, rough: Path, out: Path) -> None:
    """Fall rate with each channel pinned to a constant, both terrains.

    Not deletion: the policy net has a fixed 48-input layer, so the slots stay
    and are overwritten -- gyro obs[3:6] to (0, 0, 0), gravity obs[6:9] to
    (0, 0, -1). Both are legal in-distribution values, so what is measured is
    loss of information rather than an out-of-range input. The title says
    "held at a constant" and not "removed" for exactly that reason: the 3 %
    flat-gravity fall rate is low because the substituted upright prior is
    CORRECT on flat ground, which is unreadable if the reader thinks the
    input is gone.

    Fall rate carries the bar and return retained drops to the label, the
    reverse of the first version. Falling over is the failure a reader
    already has intuition for, and it needs no normalisation to be compared
    across terrains -- where return does, since the two policies earn
    different baselines.

    The two terrains have DIFFERENT unperturbed fall rates (0.00 flat, 0.17
    rough), so both references are drawn. Without them the rough bars read as
    ablation damage when 17 points of it is the ground.
    """
    df = np.load(flat, allow_pickle=True)
    dr = np.load(rough, allow_pickle=True)
    keys = ["ablate_gyro", "ablate_gravity", "ablate_both"]
    labels = ["gyro\nheld at zero", "gravity\nheld upright",
              "both\nheld constant"]

    # Last sweep point is the full blend (weight 1.0).
    def fell(d, k):
        return float(d[f"{k}__fell"][-1])

    def frac(d, k):
        return float(d[f"{k}__return"][-1]) / float(d["baseline__return"])

    series = (("flat terrain", BLUE, df), ("rough terrain", ORANGE, dr))
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.845))
    _style(ax)

    for i, (name, color, d) in enumerate(series):
        n = int(d["episodes"])
        falls = [fell(d, k) for k in keys]
        rets = [frac(d, k) for k in keys]
        lo, hi = wilson(np.array(falls), n)
        # 2 px surface gap between adjacent bars, per the mark spec.
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), falls, w, color=color,
                      zorder=3, label=name, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x + (i - 0.5) * (w + 0.02), falls,
                    yerr=[np.array(falls) - lo, hi - np.array(falls)],
                    fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=4,
                    capthick=1.1, zorder=6)
        for b, f, r, h in zip(bars, falls, rets, hi):
            # Inside labels sit at the BASE of the bar, not under its top:
            # the Wilson whisker spans [lo, hi] around the top and would run
            # straight through a top-anchored label. A short bar has no room,
            # so its label goes above the whisker instead.
            inside = f > 0.30
            ax.text(b.get_x() + b.get_width() / 2,
                    0.025 if inside else h + 0.025,
                    f"{f:.0%} falls\n{r:.0%} return",
                    ha="center", va="bottom",
                    color=SURFACE if inside else INK_2,
                    fontsize=8.5, linespacing=1.3, zorder=7,
                    # The short bar's label sits near the rough reference
                    # line; an opaque patch stops the dashes crossing it.
                    bbox=None if inside else dict(facecolor=SURFACE,
                                                  edgecolor="none", pad=1.5))

    # Unperturbed fall rate, one per terrain -- they are not the same.
    for name, color, d in series:
        base = float(d["baseline__fell"])
        ax.axhline(base, color=color, ls=(0, (4, 3)), lw=1.2, zorder=2)
        ax.text(2.70, base, f"{name},\nunperturbed {base:.0%}", color=color,
                fontsize=8, va="center", ha="left", linespacing=1.4)

    ax.set_xticks(x)
    for t in ax.set_xticklabels(labels, color=INK_2, fontsize=9.5,
                                linespacing=1.4):
        # Descenders differ per label ("gyro" against "both"), so bounding-box
        # alignment staggers the rows visibly. Anchor on the top instead.
        t.set_verticalalignment("top")
    ax.set_ylabel("fall rate  (error bars: Wilson 95 %)", color=INK_2,
                  fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.6, 3.38)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2,
              loc="upper left", ncols=2)

    fig.text(0.008, 0.985,
             "Fall rate and return retained with an IMU observation channel\n"
             "held at a constant (gyro at zero, gravity at nominal upright),\n"
             "flat against rough terrain, 64 episodes",
             color=INK, fontsize=12, fontweight="bold", va="top",
             linespacing=1.4)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"  {out}")


def error_budget(out: Path) -> None:
    """In-field tilt contributions against the measured degradation onset.

    The onset is command-conditional (section 4.2), so the figure carries the
    range across motion regimes rather than the single in-motion number an
    earlier version drew. Thresholds are the project-wide 5 % / 10 % return
    loss from scripts/knees.py, not read off the sweep grid by eye.
    """
    from imu_diff_sim.validation import bmi270_spec as S

    dT = 40.0
    terms = [
        ("accel zero-g offset\n(±20 mg, datasheet)", S.ACC_ZERO_G_OFFSET_MG * 1e-3),
        (f"accel TCO drift\n(0.25 mg/K × {dT:.0f} K)", S.ACC_TCO_MG_PER_K * dT * 1e-3),
        ("accel cross-axis\n(1 %, datasheet)", S.ACC_CROSS_AXIS),
        (f"differential TCS\n(0.004 %/K × {dT:.0f} K)", S.ACC_TCS_PER_K * dT),
    ]
    deg = lambda v: np.degrees(np.arctan(v))
    vals = [deg(v) for _, v in terms]
    rss = deg(np.sqrt(sum(v ** 2 for _, v in terms)))
    mount = 3.0

    labels = [t for t, _ in terms] + ["RSS of\ndatasheet terms",
                                      "+ 3° mechanical\nmounting error"]
    heights = vals + [rss, rss + mount]
    colors = [BLUE] * 4 + [AQUA, ORANGE]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.885))
    _style(ax)
    bars = ax.bar(range(len(heights)), heights, 0.62, color=colors, zorder=3,
                  edgecolor=SURFACE, linewidth=2)
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.12, f"{h:.2f}°",
                ha="center", va="bottom", color=INK_2, fontsize=9)

    # The 5 % onset is not one number: it varies with what the robot was asked
    # to do. Drawing the range rather than a single line is the honest form --
    # the stack-up sits INSIDE it in every regime, which is the argument.
    # Values from scripts/knees.py; see reports/figures/ofat-command-regimes.png.
    onset_lo, onset_hi, knee = 4.10, 5.07, 5.14
    ax.axhspan(onset_lo, onset_hi, color=ORANGE, alpha=0.12, zorder=1)
    for y, lab in ((onset_hi, "5 % onset, standing still 5.07°  "),
                   (onset_lo, "5 % onset, forward 1.5 m/s 4.10°  ")):
        ax.axhline(y, color=ORANGE, ls=(0, (5, 3)), lw=1.6, zorder=4)
        ax.text(-0.42, y, lab, color=ORANGE, fontsize=8.5, va="bottom",
                ha="left")
    ax.axhline(knee, color=INK_MUTED, ls=":", lw=1.3, zorder=4)
    # Right-anchored: the knee is 0.07 deg above the standing onset, so a
    # left-anchored label lands on top of that one.
    ax.text(len(heights) - 0.55, knee,
            f"  10 % knee, sampled command {knee:.2f}°",
            color=INK_MUTED, fontsize=8.5, va="bottom", ha="right")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK_2, fontsize=8.5, linespacing=1.4)
    ax.set_ylabel("attitude error [degrees]", color=INK_2, fontsize=9.5)
    ax.set_ylim(0, 6.4)
    fig.text(0.008, 0.985,
             "In-field attitude error terms and their stack-up, against\n"
             "the measured 5 % onset and 10 % knee in gravity tilt",
             color=INK, fontsize=12, fontweight="bold", va="top",
             linespacing=1.4)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"  {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat", type=Path, default=Path("runs/ofat_go1_clean_abl.npz"))
    ap.add_argument("--rough", type=Path, default=Path("runs/ofat_rough_abl.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("wrote:")
    ablation(args.flat, args.rough, args.outdir / "ablation.png")
    error_budget(args.outdir / "error-budget.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
