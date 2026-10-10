import cv2
import numpy as np

FILL = (114, 114, 114)


def rectify(image: np.ndarray, quad: np.ndarray, size: int = 256, fill=FILL):
    """Warp a planogram quad to a fronto-parallel rectangle, letterboxed into a size x size square.

    The rectangle keeps the area's native aspect ratio. Letterbox padding and any part of the
    area outside the camera frame are set to `fill`.

    Returns (crop, valid_mask, visible_frac, (native_w, native_h)), where valid_mask is True
    for pixels sampled from inside the camera image and visible_frac is the share of the
    area that lies inside the frame.
    """
    tl, tr, br, bl = quad
    native_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    native_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    aspect = native_w / native_h
    tw, th = (size, max(1, round(size / aspect))) if aspect >= 1 else (max(1, round(size * aspect)), size)
    ox, oy = (size - tw) // 2, (size - th) // 2
    dst = np.array([[ox, oy], [ox + tw, oy], [ox + tw, oy + th], [ox, oy + th]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    crop = cv2.warpPerspective(image, H, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    in_frame = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), H, (size, size),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0

    rect = np.zeros((size, size), bool)
    rect[oy:oy + th, ox:ox + tw] = True
    valid = in_frame & rect
    # warpPerspective samples the image beyond the quad into the padding; blank it out.
    crop[~valid] = fill

    return crop, valid, float(valid.sum() / rect.sum()), (round(native_w), round(native_h))
