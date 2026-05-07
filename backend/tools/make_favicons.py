#!/usr/bin/env python3
"""Generate favicon variants from a single source image.

Output (in tutorials/imgs/):
  favicon-16.png         16x16  (browser tab)
  favicon-32.png         32x32  (browser tab, hi-dpi)
  apple-touch-icon.png   180x180 (iOS home screen)
  favicon.ico            multi-size (16+32) for legacy browsers

Source priority (first existing wins):
  1. rag_pet_face.png  — preferred. Cropped head/face, sharp at small sizes.
  2. rag_pet.png       — fallback. Used until the cropped version is dropped in.

Usage:
    uv run python -m tools.make_favicons
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

IMGS_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "imgs"

CANDIDATES = ["rag_pet_face.png", "rag_pet.png"]
SIZES = {
    "favicon-16.png": (16, 16),
    "favicon-32.png": (32, 32),
    "apple-touch-icon.png": (180, 180),
}


def pick_source() -> Path:
    for name in CANDIDATES:
        p = IMGS_DIR / name
        if p.exists():
            return p
    print("error: no source image found. Drop rag_pet.png or rag_pet_face.png in", IMGS_DIR, file=sys.stderr)
    raise SystemExit(2)


def to_square(img: Image.Image) -> Image.Image:
    """Crop to a centered square (vertical center is biased toward the top
    third — assumes the subject's face/head sits there in a full-body shot)."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    if w > h:
        left = (w - side) // 2
        top = 0
    else:
        left = 0
        # bias upward — heads tend to be in the upper portion of full-body shots
        top = max(0, (h - side) // 4)
    return img.crop((left, top, left + side, top + side))


def main() -> int:
    src = pick_source()
    print(f"source: {src.name}")

    img = Image.open(src).convert("RGBA")
    img = to_square(img)

    for fname, size in SIZES.items():
        resized = img.resize(size, Image.LANCZOS)
        out = IMGS_DIR / fname
        resized.save(out, format="PNG", optimize=True)
        print(f"  wrote {out.name}  ({size[0]}x{size[1]})")

    # Multi-size .ico for legacy browsers.
    ico_path = IMGS_DIR / "favicon.ico"
    img.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print(f"  wrote {ico_path.name}  (multi-size ICO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
