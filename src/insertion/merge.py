import cv2
import numpy as np

from src.config import get_config
from src.utils import normalize_rect


def get_block_points(block: dict) -> np.ndarray | None:
    polygons = block.get("poly_points", [])
    pts = [pt for poly in polygons for pt in poly]
    return np.array(pts, dtype=np.int32) if pts else None


def block_shape_hull(block: dict) -> np.ndarray | None:
    pts = get_block_points(block)
    if pts is None or len(pts) < 3:
        return None
    return cv2.convexHull(pts)


def _compute_rotation(
    pts: np.ndarray, threshold: float
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    if len(pts) < 3:
        return 0.0, (0.0, 0.0), (0, 0)

    center, (w, h), angle = normalize_rect(pts)

    if abs(angle) <= threshold:
        angle = 0.0

    return angle, center, (int(round(w)), int(round(h)))


def merge_detections(
    img: np.ndarray,
    blocks: list[dict],
) -> list[dict]:
    config = get_config()
    merged_blocks = []
    for block in blocks:
        hull = block_shape_hull(block)
        if hull is None:
            continue

        block_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(block_mask, [hull], 255)

        pts = get_block_points(block)
        angle, center, rect_size = _compute_rotation(pts, config.text_angle_threshold)

        merged_blocks.append(
            {
                "mask": block_mask,
                "text": block["translated_text"],
                "angle": angle,
                "rect_center": center,
                "rect_size": rect_size,
            }
        )
    return merged_blocks
