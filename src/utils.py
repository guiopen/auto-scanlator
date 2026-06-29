import cv2
import numpy as np


def normalize_rect(
    pts: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    rect = cv2.minAreaRect(pts)
    center, (w, h), angle = rect
    if angle < -45:
        angle += 90
        w, h = h, w
    return center, (w, h), angle
