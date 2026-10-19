"""Two figures the write-up needs and the sweep plots do not cover.

1. Ablation: removing a channel versus corrupting it, flat against rough.
   The most quotable result in the project had no figure.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    """Return retained after fully removing each channel, both terrains."""
    df, dr = np.load(flat, allow_pickle=True), np.load(rough, allow_pickle=True)
    keys = ["ablate_gyro", "ablate_gravity", "ablate_both"]
    labels = ["gyro\nremoved", "gravity\nremoved", "both\nremoved"]

    def frac(d, k):
        # Last sweep point is full removal (blend weight 1.0).
        return float(d[f"{k}__return"][-1]) / float(d["baseline__return"])

    vals = {"flat terrain": [frac(df, k) for k in keys],
            "rough terrain": [frac(dr, k) for k in keys]}
    falls = {"flat terrain": [float(df[f"{k}__fell"][-1]) for k in keys],
             "rough terrain": [float(dr[f"{k}__fell"][-1]) for k in keys]}

    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    _style(ax)

    for i, (name, color) in enumerate((("flat terrain", BLUE),
                                       ("rough terrain", ORANGE))):
        # 2 px surface gap between adjacent bars, per the mark spec.
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals[name], w, color=color,
                      zorder=3, label=name, edgecolor=SURFACE, linewidth=2)
        for b, v, f in zip(bars, vals[name], falls[name]):
            # Label inside the bar when it is tall enough to hold the text,
            # otherwise above it -- avoids colliding with the 100% reference.
            inside = v > 0.28
            ax.text(b.get_x() + b.get_width() / 2,
                    v - 0.03 if inside else v + 0.022,
                    f"{v:.0%}\n{f:.0%} falls", ha="center",
                    va="top" if inside else "bottom",
                    color=SURFACE if inside else INK_2,
                    fontsize=8.5, linespacing=1.3, zorder=5)

    ax.axhline(1.0, color=INK_MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(-0.56, 1.0, "unperturbed ", color=INK_MUTED, fontsize=8.5,
            va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK_2, fontsize=9.5)
    ax.set_ylabel("return retained", color=INK_2, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.6, 2.6)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2,
              loc="upper right", ncols=2)

    fig.text(0.008, 0.985,
             "Removing a channel is not the same as corrupting it",
             color=INK, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.008, 0.935,
             "The gyro is immune to 635x its datasheet noise, yet essential. "
             "Gravity is the reverse: nearly free to lose on flat\nground, "
             "ruinous on rough -- because the upright prior it falls back on "
             "is only correct when the ground is flat.",
             color=INK_2, fontsize=8.5, va="top", linespacing=1.5)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"  {out}")


def error_budget(out: Path) -> None:
    """In-field tilt contributions against the measured degradation onset."""
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
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    _style(ax)
    bars = ax.bar(range(len(heights)), heights, 0.62, color=colors, zorder=3,
                  edgecolor=SURFACE, linewidth=2)
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.12, f"{h:.2f}°",
                ha="center", va="bottom", color=INK_2, fontsize=9)

    ax.axhline(4.54, color=ORANGE, ls=(0, (5, 3)), lw=1.6, zorder=4)
    ax.text(-0.42, 4.54, "measured degradation onset 4.54°  ", color=ORANGE,
            fontsize=8.5, va="bottom", ha="left")
    ax.axhline(11.28, color=INK_MUTED, ls=":", lw=1.3, zorder=4)
    ax.text(-0.42, 11.28, "measured 10 % knee 11.28°  ", color=INK_MUTED,
            fontsize=8.5, va="bottom", ha="left")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK_2, fontsize=8.5, linespacing=1.4)
    ax.set_ylabel("attitude error [degrees]", color=INK_2, fontsize=9.5)
    ax.set_ylim(0, 12.6)
    fig.text(0.008, 0.985,
             "A realistic field stack-up lands on the failure threshold",
             color=INK, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.008, 0.925,
             "Gravity tilt is a DIRECTION error: the vector is renormalised, "
             "so a common-mode scale error contributes nothing.\nThe terms "
             "that do contribute are per-axis, persistent and slowly varying "
             "-- and the largest is mechanical, not a datasheet line.",
             color=INK_2, fontsize=8.5, va="top", linespacing=1.5)
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
