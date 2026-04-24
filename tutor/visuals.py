"""Render simple math visuals (counting) without external model weights."""

from __future__ import annotations

import io
import random
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

ObjectLabel = Literal["goat", "circle", "dot"]


def _safe_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_count_image(
    n_objects: int,
    object_label: str = "circle",
    seed: int | None = None,
    size: tuple[int, int] = (512, 384),
    *,
    with_caption: bool = False,
) -> bytes:
    """Return PNG bytes: n child-friendly pictorials in a simple grid. No on-image text by default
    (``with_caption=False``) for pre-literate UIs.
    """
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", (w, h), color=(250, 248, 240))
    draw = ImageDraw.Draw(img)
    label = object_label.lower()
    colors = [
        (240, 100, 100),
        (80, 160, 240),
        (100, 200, 120),
        (230, 180, 60),
        (180, 120, 220),
    ]
    cols = min(6, max(3, int(n_objects**0.5) + 1))
    rows = max(1, (n_objects + cols - 1) // cols)
    cell_w, cell_h = w // (cols + 1), h // (rows + 2)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n_objects:
                break
            cx = (c + 1) * cell_w
            cy = (r + 1) * cell_h
            jitter = rng.randint(-6, 6)
            color = colors[count % len(colors)]
            if label == "goat":
                _draw_goat(draw, cx + jitter, cy, min(cell_w, cell_h) // 2, color)
            elif label == "dot":
                draw.ellipse(
                    [cx - 10, cy - 10, cx + 10, cy + 10],
                    fill=color,
                    outline=(40, 40, 40),
                    width=1,
                )
            elif label == "finger":
                _draw_finger(draw, cx + jitter, cy, min(cell_w, cell_h) // 3, color)
            else:
                draw.ellipse(
                    [cx - 16, cy - 16, cx + 16, cy + 16],
                    fill=color,
                    outline=(60, 60, 60),
                    width=2,
                )
            count += 1
    if with_caption:
        font = _safe_font(16)
        draw.text((12, h - 28), f"Count the {label}s", fill=(50, 50, 50), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _draw_finger(
    draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: tuple[int, int, int]
) -> None:
    """Simple upright 'finger' pill for counting-to-five style prompts."""
    body = (cx - r // 2, cy - 2 * r, cx + r // 2, cy + r)
    draw.rounded_rectangle(body, radius=r // 3, fill=color, outline=(50, 50, 50), width=2)


def _draw_goat(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    body = (cx - r, cy - r // 2, cx + r, cy + r)
    draw.ellipse(body, fill=color, outline=(40, 40, 40), width=2)
    head = (cx + r - 4, cy - r, cx + r + 10, cy - r + 12)
    draw.ellipse(head, fill=(240, 220, 200), outline=(40, 40, 40), width=1)
    leg = (cx - r // 2, cy + r - 2, cx - r // 2 + 4, cy + r + 8)
    leg2 = (cx + r // 2 - 4, cy + r - 2, cx + r // 2, cy + r + 8)
    draw.rectangle(leg, fill=(50, 50, 50))
    draw.rectangle(leg2, fill=(50, 50, 50))
