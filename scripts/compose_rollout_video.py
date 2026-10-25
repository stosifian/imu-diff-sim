"""Stitch the flat and rough rollouts into one labelled clip for the write-up.

Two panes side by side under caption strips, so the reader sees in one asset
that the same task was measured on both terrains. The strips are drawn rather
than burned into the frame because the two scenes have opposite brightness --
white floor against a near-black heightfield -- and no single text colour
reads on both.

Emits .mp4 (quality) and .gif (GitHub renders a relative-path GIF inline; it
will not play a relative-path mp4).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

SURFACE, INK = (252, 252, 251), (11, 11, 11)
STRIP_H, GAP = 34, 4
FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _captioned(frames: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for path in FONTS:
        if Path(path).exists():
            font = ImageFont.truetype(path, 19)
            break
    font = font or ImageFont.load_default()

    strip = Image.new("RGB", (frames.shape[2], STRIP_H), SURFACE)
    draw = ImageDraw.Draw(strip)
    draw.text((12, STRIP_H // 2), text, font=font, fill=INK, anchor="lm")
    band = np.tile(np.asarray(strip)[None], (len(frames), 1, 1, 1))
    return np.concatenate([band, frames], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--left-label", default="flat terrain")
    ap.add_argument("--right-label", default="rough terrain")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/figures/go1-rollout.mp4"))
    ap.add_argument("--crop-bottom", type=int, default=70,
                    help="rows of empty floor to trim; the track camera "
                         "frames the robot high in a 4:3 render")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-colors", type=int, default=64)
    args = ap.parse_args()

    import mediapy

    left, right = mediapy.read_video(args.left), mediapy.read_video(args.right)
    fps = left.metadata.fps
    n = min(len(left), len(right))
    keep = slice(None, -args.crop_bottom or None)
    panes = [
        _captioned(np.asarray(left)[:n, keep], args.left_label),
        _captioned(np.asarray(right)[:n, keep], args.right_label),
    ]
    gap = np.full((n, panes[0].shape[1], GAP, 3), SURFACE, dtype=np.uint8)
    out = np.concatenate([panes[0], gap, panes[1]], axis=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(args.out, out, fps=fps)
    print(f"{n} frames, {out.shape[2]}x{out.shape[1]} at {fps:.0f} fps -> {args.out}")

    # Two-pass palette: a global palette on this many frames is both smaller
    # and cleaner than ffmpeg's default per-frame quantisation.
    gif = args.out.with_suffix(".gif")
    vf = (f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,"
          f"split[a][b];[a]palettegen=max_colors={args.gif_colors}[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(args.out), "-vf", vf,
         "-loop", "0", str(gif)], check=True
    )
    print(f"{gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
