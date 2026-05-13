"""Generate a 1200x630 Open Graph banner for the tutorials landing page.

The other tutorial pages have AI-generated banners; the landing page didn't,
so LinkedIn / Twitter would show a small square thumbnail instead of a
proper landscape card. This composes one programmatically by placing the
existing mascot illustration on a dark canvas with the site title.

Run from backend/:
    uv run python -m tools.make_og_index
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
IMGS = ROOT / "tutorials" / "imgs"
OUT = IMGS / "og_index.png"

W, H = 1200, 630
BG = (10, 18, 32)          # near-black navy, matches existing OG palette
PURPLE = (167, 105, 220)   # mascot purple / "RAG" accent
WHITE = (245, 243, 238)
MUTED = (170, 175, 190)

FONT_BOLD = "/System/Library/Fonts/HelveticaNeue.ttc"     # index 1 = Bold
FONT_REG = "/System/Library/Fonts/HelveticaNeue.ttc"      # index 0 = Regular


def _font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


def _vertical_gradient(size: tuple[int, int], top: tuple[int, int, int],
                       bottom: tuple[int, int, int]) -> Image.Image:
    """Cheap top→bottom linear gradient as a base layer."""
    w, h = size
    grad = Image.new("RGB", size, top)
    px = grad.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return grad


def _fit_mascot(target_h: int) -> Image.Image:
    """Load the brewing-octopus mascot and scale it to a given height."""
    mascot = Image.open(IMGS / "rag_pet.png").convert("RGBA")
    ratio = target_h / mascot.height
    new_w = int(mascot.width * ratio)
    return mascot.resize((new_w, target_h), Image.LANCZOS)


def main() -> None:
    canvas = _vertical_gradient((W, H), top=(14, 22, 38), bottom=(6, 12, 22))

    # Mascot on the right, bleeding a little off the right edge.
    mascot = _fit_mascot(target_h=H + 40)
    mx = W - mascot.width + 60
    my = -20
    canvas.paste(mascot, (mx, my), mascot)

    # Soft dark vignette over the mascot so the typography stays legible
    # even where it overlaps the illustration.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, 0), (W // 2 + 120, H)], fill=(8, 14, 26, 180))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(canvas)

    # Purple accent bar
    draw.rectangle([(60, 78), (130, 90)], fill=PURPLE)

    # Eyebrow
    draw.text(
        (60, 100),
        "TUTORIALS · GENAI · PRODUCTION-GRADE",
        font=_font(FONT_BOLD, 20, index=1),
        fill=MUTED,
    )

    # Title — two stacked lines, with the accent on "RAG". Condensed Black
    # is the heaviest face in HelveticaNeue.ttc and reads as a magazine title.
    title_font = _font(FONT_BOLD, 110, index=9)  # Condensed Black
    draw.text((60, 150), "THE ANNOTATED", font=title_font, fill=WHITE)
    draw.text((60, 280), "RAG", font=title_font, fill=PURPLE)

    # Tagline
    tag_font = _font(FONT_REG, 26, index=0)
    draw.text(
        (60, 430),
        "Tutorials by reading a real production RAG —",
        font=tag_font, fill=WHITE,
    )
    draw.text(
        (60, 466),
        "framework-agnostic concepts + two engines, side by side.",
        font=tag_font, fill=WHITE,
    )

    # Language pill
    pill_font = _font(FONT_BOLD, 22, index=1)
    pill_text = "EN · ES"
    pad_x, pad_y = 18, 10
    # textbbox of UPPERCASE is more reliable than getbbox on the font.
    bbox = pill_font.getbbox(pill_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ascent = -bbox[1]  # how far above baseline the glyphs go
    px, py = 60, 525
    pill_h = th + 2 * pad_y
    draw.rounded_rectangle(
        [(px, py), (px + tw + 2 * pad_x, py + pill_h)],
        radius=999, fill=PURPLE,
    )
    # Center text vertically inside the pill.
    text_y = py + (pill_h - th) // 2 - ascent + 1
    draw.text((px + pad_x, text_y), pill_text, font=pill_font, fill=(20, 12, 32))

    canvas.convert("RGB").save(OUT, format="PNG", optimize=True)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
