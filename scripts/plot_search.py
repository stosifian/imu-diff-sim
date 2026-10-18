"""The money plot: uniform DR vs CEM over IMU mounting misalignment.

Form. Two questions, two panels, never a shared axis. Left: where in the
parameter space each arm spent its budget, which is a coverage question and
wants a scatter. Right: failures discovered against budget, which is a
cumulative-rate question and wants lines.

Colour. Categorical slots 1 and 2 of the validated palette in fixed order,
one per arm, directly labelled as well as in the legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
ARMS = (("uniform", BLUE, "uniform DR"), ("cem", ORANGE, "CEM"))


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/figures/search-misalign.png"))
    args = ap.parse_args()
    d = np.load(args.npz)
    base, thresh = float(d["baseline"]), float(d["threshold"])
    bound_deg = np.degrees(float(d["bound_rad"]))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.4),
                                   constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))

    # Left: total rotation magnitude vs return. The 3-D box projected onto the
    # quantity that physically matters -- total angular error -- rather than
    # an arbitrary pair of axes.
    _style(axL)
    for key, color, label in ARMS:
        p, v = d[f"{key}__points"], d[f"{key}__returns"]
        mag = np.degrees(np.linalg.norm(p, axis=1))
        axL.scatter(mag, v, s=14, color=color, alpha=0.55, lw=0, label=label,
                    zorder=3)
    axL.axhline(thresh, color=INK_MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)
    axL.text(0.015, thresh, f" failure threshold ({int(float(d['fail_fraction'])*100)}% of baseline)",
             transform=axL.get_yaxis_transform(), color=INK_MUTED, fontsize=7.5,
             va="bottom")
    axL.axvline(4.54, color=INK_MUTED, ls=":", lw=1.1, zorder=2)
    axL.text(4.54, 0.02, " OFAT degradation onset", rotation=90, fontsize=7.5,
             color=INK_MUTED, transform=axL.get_xaxis_transform(), va="bottom")
    axL.set_xlabel("total mounting misalignment [deg]", color=INK_2, fontsize=9)
    axL.set_ylabel("return", color=INK_2, fontsize=9)
    axL.set_title("Where each arm spent its budget", color=INK, fontsize=10.5,
                  fontweight="bold", loc="left", pad=6)
    axL.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")

    # Right: cumulative failures against evaluations.
    _style(axR)
    for key, color, label in ARMS:
        v = d[f"{key}__returns"]
        cum = np.cumsum(v < thresh)
        axR.plot(np.arange(1, len(v) + 1), cum, color=color, lw=2.0, zorder=4,
                 label=label)
        axR.annotate(f"{label}: {cum[-1]}", (len(v), cum[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=color, fontsize=9, va="center")
    axR.set_xlabel("parameter evaluations (equal budget per arm)",
                   color=INK_2, fontsize=9)
    axR.set_ylabel("failures found (cumulative)", color=INK_2, fontsize=9)
    axR.set_title("Discovery rate", color=INK, fontsize=10.5,
                  fontweight="bold", loc="left", pad=6)
    axR.set_xlim(0, len(d["cem__returns"]) * 1.22)
    axR.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    fig.text(0.008, 0.985,
             "Adaptive search finds many more failures -- but not worse ones",
             color=INK, fontsize=12.5, fontweight="bold", va="top")
    worst = {k: float(np.min(d[f"{k}__returns"])) for k, _, _ in ARMS}
    fig.text(0.008, 0.945,
             f"box +-{bound_deg:.0f} deg/axis  |  baseline {base:.2f}  |  "
             f"worst found: uniform {worst['uniform']:.2f}, CEM {worst['cem']:.2f}"
             f"  |  CEM collapsed to a single mode (sigma 0.07 deg)",
             color=INK_2, fontsize=8.5, va="top")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
