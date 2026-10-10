"""Handcrafted per-crop measurements, computed only on real shelf pixels (the area's valid mask).

Group A, "how much empty container is visible": empty containers are teal crates, grey/metal trays
or black trays, while produce is mostly saturated colour.
Group B, "how busy is the image": piles of produce are full of outlines and texture whatever their
colour, while empty trays and shelf covers are smooth.
"""
import cv2
import numpy as np

from shelfmon.planogram import Area
from shelfmon.rectify import rectify

# OpenCV HSV ranges: hue 0-179, saturation and value 0-255. Thresholds set by looking at empty
# crates and trays in camera 6462 at 06:12 (before restocking).
TEAL_HUE = (75, 100)
TEAL_MIN_SAT = 90
LOW_COLOUR_MAX_SAT = 70  # grey/metal trays look slightly blue in shadow, still below this
DARK_MAX_VALUE = 50

CONTAINER_FEATURES = ["teal_frac", "low_colour_frac", "dark_frac", "mean_saturation", "colourfulness"]
TEXTURE_FEATURES = ["edge_frac", "edge_frac_coarse", "laplacian"]
CANNY_THRESHOLDS = (50, 150)
TEXTURE_EXTRA_ERODE = 3  # the step from shelf pixels to grey padding is itself an edge; stay clear of it


def area_mask(area: Area, frame_size: tuple[int, int], size: int = 256, erode: int = 2) -> np.ndarray:
    """Boolean mask of real shelf pixels in the area's crop (no letterbox padding, no off-frame parts).

    It depends only on the area geometry, so it is the same for every image of a camera. Eroded a little
    so JPEG blur at the padding edge isn't counted.
    """
    width, height = frame_size
    _, valid, _, _ = rectify(np.zeros((height, width, 3), np.uint8), area.quad, size)
    mask = valid.astype(np.uint8)
    if erode:
        mask = cv2.erode(mask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return mask.astype(bool)


def container_features(crop_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Group A: share of teal-crate, low-colour and dark pixels, plus how colourful the crop is."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[mask].astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    teal = (h >= TEAL_HUE[0]) & (h <= TEAL_HUE[1]) & (s >= TEAL_MIN_SAT) & (v > DARK_MAX_VALUE)

    # Hasler & Suesstrunk colourfulness: spread and strength of the opponent colour channels.
    b, g, r = crop_bgr[mask].astype(np.float32).T
    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    return {
        "teal_frac": float(teal.mean()),
        "low_colour_frac": float((s < LOW_COLOUR_MAX_SAT).mean()),
        "dark_frac": float((v < DARK_MAX_VALUE).mean()),
        "mean_saturation": float(s.mean() / 255),
        "colourfulness": float(colourfulness / 255),
    }


def texture_features(crop_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Group B: share of edge pixels (full and half resolution) and strength of fine texture."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * TEXTURE_EXTRA_ERODE + 1,) * 2, np.uint8)).astype(bool)
    if not inner.any():
        inner = mask
    gray = cv2.GaussianBlur(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, *CANNY_THRESHOLDS) > 0

    # Half resolution keeps fruit outlines but drops tiny detail such as crate holes and JPEG noise.
    half = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    half_inner = cv2.resize(inner.astype(np.uint8), half.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    edges_coarse = cv2.Canny(half, *CANNY_THRESHOLDS) > 0

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)[inner]
    return {
        "edge_frac": float(edges[inner].mean()),
        "edge_frac_coarse": float(edges_coarse[half_inner].mean()) if half_inner.any() else 0.0,
        "laplacian": float(np.log1p(laplacian.var())),
    }


AB_FEATURES = CONTAINER_FEATURES + TEXTURE_FEATURES


def crop_features(crop_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Groups A and B for one crop."""
    return container_features(crop_bgr, mask) | texture_features(crop_bgr, mask)
