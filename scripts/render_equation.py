"""Render the overlapping Allan deviation estimator as a figure.

Transcribed from imu_diff_sim.allan.overlapping_allan_deviation -- if that
implementation changes, this must be re-rendered. Uses matplotlib's mathtext
rather than a LaTeX install, so it runs anywhere the rest of the repo does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"

# theta is the integrated signal; the estimator is a second difference of it.
INTEGRAL = r"$\theta_k \;=\; \Delta t \sum_{i<k} \left( \Omega_i - \bar{\Omega} \right)$"
ESTIMATOR = (
    r"$\sigma^2(\tau) \;=\; \dfrac{1}{2\,\tau^2\,(N - 2m + 1)}"
    r"\sum_{k=0}^{N-2m} \left( \theta_{k+2m} - 2\theta_{k+m} + \theta_k \right)^2$"
)


def main() -> int:
    fig = plt.figure(figsize=(8.6, 3.5))
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.5, 0.90, "Overlapping Allan variance, as implemented",
             ha="center", va="top", color=INK, fontsize=12, fontweight="bold")

    fig.text(0.5, 0.715, INTEGRAL, ha="center", va="center",
             color=INK, fontsize=15)
    fig.text(0.5, 0.60,
             "integrate the mean-removed rate signal (angle, for a gyro)",
             ha="center", va="center", color=INK_2, fontsize=8.5)

    fig.text(0.5, 0.385, ESTIMATOR, ha="center", va="center",
             color=INK, fontsize=15)
    fig.text(0.5, 0.235,
             "second difference over a window of m samples, "
             r"with $\tau = m\,\Delta t$;  Allan deviation is $\sigma(\tau)$",
             ha="center", va="center", color=INK_2, fontsize=8.5)

    fig.text(0.5, 0.085,
             "Overlapping, not block: every starting index k contributes, so "
             "the long-$\\tau$ end\nretains usable statistics where the "
             "random-walk slope is read.",
             ha="center", va="center", color=INK_2, fontsize=8, linespacing=1.6)

    out = Path("reports/figures/allan-equation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
