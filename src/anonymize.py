# src/anonymize.py — blur / pixelate / solid, with box padding and elliptical mask for faces
import cv2
import numpy as np


def expand(box, pad, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    x2, y2 = int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
    return x1, y1, x2, y2


def anonymize(frame, boxes, method="blur", pad=0.25, ellipse=True):
    H, W = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = expand(box, pad, W, H)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        roi = frame[y1:y2, x1:x2]

        if method == "blur":
            sw, sh = max(1, w // 8), max(1, h // 8)                    # blur at 1/8 scale: same look, much cheaper
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            k = max(3, (max(sw, sh) // 2) | 1)
            small = cv2.GaussianBlur(small, (k, k), 0)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        elif method == "pixelate":
            small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif method == "solid":
            out = np.zeros_like(roi)
        else:
            raise ValueError(method)

        if ellipse:
            mask = np.zeros((h, w), np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
            roi[:] = np.where(mask[..., None] > 0, out, roi)
        else:
            roi[:] = out
    return frame