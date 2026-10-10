import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shelfmon.planogram import Area

# Pillow's bundled font lacks glyphs such as Ã, so prefer a system font with full Latin coverage.
FONT_CANDIDATES = ["LiberationSans-Regular.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf",
                   "/System/Library/Fonts/Supplemental/Arial.ttf"]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


FONT_ID = _load_font(15)
FONT_NAME = _load_font(12)
FONT_LEGEND = _load_font(13)
HAS_SYSTEM_FONT = FONT_ID.path != ImageFont.load_default(size=15).path
OUTLINE = (255, 255, 0)
PAD = 4
STRIPE = 4
# Bright colours that stay visible on dark shelves and light floors.
PALETTE = [(255, 225, 25), (0, 230, 255), (255, 80, 200), (120, 255, 60), (255, 140, 0), (255, 255, 255),
           (255, 70, 70), (160, 130, 255), (0, 255, 170), (255, 180, 200), (190, 240, 120), (90, 170, 255)]


def product_colours(areas: list[Area]) -> dict[str, tuple[int, int, int]]:
    """RGB colour per product id; areas of the same product share a colour."""
    ids = list(dict.fromkeys(a.product_id for a in areas))
    return {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(ids)}


def visible_polygon(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    """Part of the area quad that lies inside the camera frame."""
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    _, inter = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    return inter.reshape(-1, 2) if inter is not None else quad


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float, max_lines: int = 2) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        candidate = f"{cur} {word}".strip()
        if not cur or draw.textlength(candidate, font=font) <= max_w:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        last = lines[max_lines - 1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines = lines[:max_lines - 1] + [last.rstrip() + "…"]
    return lines


def _text(s: str) -> str:
    """Strip accents when falling back to Pillow's bundled font, which can't render all of them."""
    if HAS_SYSTEM_FONT:
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _draw_outlines(draw: ImageDraw.ImageDraw, areas: list[Area], width: int):
    for area in areas:
        draw.polygon([tuple(p) for p in area.quad.tolist()], outline=OUTLINE, width=width)


def draw_boxes(image_bgr: np.ndarray, areas: list[Area], width: int = 3) -> np.ndarray:
    """Area outlines only."""
    img = _to_pil(image_bgr)
    _draw_outlines(ImageDraw.Draw(img), areas, width)
    return _to_bgr(img)


def _label_lines(draw, area: Area, poly: np.ndarray):
    """Full label (area id + product name) if it fits inside the visible area, else just the area number."""
    vis_w, vis_h = np.ptp(poly[:, 0]), np.ptp(poly[:, 1])
    max_w = vis_w * 0.9 - 2 * PAD - STRIPE
    if max_w >= 60:
        lines = [(area.area_id, FONT_ID)] + [(t, FONT_NAME) for t in _wrap(draw, _text(area.product_name), FONT_NAME, max_w)]
        heights = [draw.textbbox((0, 0), t, font=f)[3] + 2 for t, f in lines]
        if sum(heights) + 2 * PAD <= vis_h * 0.9:
            return lines, heights
    lines = [(area.area_id.rsplit("_", 1)[1], FONT_ID)]
    return lines, [draw.textbbox((0, 0), lines[0][0], font=FONT_ID)[3] + 2]


def _draw_legend(img: Image.Image, areas: list[Area], colours) -> Image.Image:
    """Append a panel below the image listing every area with its full product name."""
    W, H = img.size
    measure = ImageDraw.Draw(img)
    entries = [f"{a.area_id}  {_text(a.product_name)}" for a in areas]
    col_w = int(max(measure.textlength(e, font=FONT_LEGEND) for e in entries)) + 28
    cols = max(1, W // col_w)
    rows = -(-len(entries) // cols)
    row_h = 19
    out = Image.new("RGBA", (W, H + rows * row_h + 2 * PAD), (25, 25, 25, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, (area, text) in enumerate(zip(areas, entries)):
        x = PAD + (i // rows) * col_w
        y = H + PAD + (i % rows) * row_h
        draw.rectangle((x, y + 3, x + 11, y + 14), fill=colours[area.product_id])
        draw.text((x + 18, y + 1), text, font=FONT_LEGEND, fill=(255, 255, 255))
    return out


def draw_annotated(image_bgr: np.ndarray, areas: list[Area], width: int = 3) -> np.ndarray:
    """Area outlines, an in-place label per area, and a legend with every full product name."""
    img = _to_pil(image_bgr).convert("RGBA")
    W, H = img.size
    colours = product_colours(areas)
    _draw_outlines(ImageDraw.Draw(img), areas, width)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    placed = []
    for area in areas:
        poly = visible_polygon(area.quad, W, H)
        m = cv2.moments(poly.astype(np.float32))
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else poly.mean(axis=0)

        lines, heights = _label_lines(draw, area, poly)
        box_w = max(draw.textlength(t, font=f) for t, f in lines) + 2 * PAD + STRIPE
        box_h = sum(heights) + 2 * PAD

        # Centre on the visible area, clamp inside the frame, then nudge vertically off earlier labels.
        x0 = int(np.clip(cx - box_w / 2, 0, W - box_w))
        for k in (0, 1, -1, 2, -2):
            y0 = int(np.clip(cy - box_h / 2 + k * (box_h + 2), 0, H - box_h))
            rect = (x0, y0, x0 + box_w, y0 + box_h)
            if not any(_overlaps(rect, r) for r in placed):
                break
        placed.append(rect)

        draw.rectangle(rect, fill=(0, 0, 0, 180))
        draw.rectangle((x0, y0, x0 + STRIPE - 1, y0 + box_h), fill=colours[area.product_id] + (255,))
        ty = y0 + PAD
        for (text, font), h in zip(lines, heights):
            draw.text((x0 + STRIPE + PAD, ty), text, font=font, fill=(255, 255, 255, 255))
            ty += h

    return _to_bgr(_draw_legend(Image.alpha_composite(img, overlay), areas, colours))
