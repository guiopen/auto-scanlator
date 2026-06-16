import cv2
import numpy as np

from src.config import get_config


def get_block_points(block: dict) -> np.ndarray | None:
    polygons = block.get("poly_points", [])
    pts = [pt for poly in polygons for pt in poly]
    return np.array(pts, dtype=np.int32) if pts else None


def block_shape_hull(block: dict) -> np.ndarray | None:
    pts = get_block_points(block)
    if pts is None or len(pts) < 3:
        return None
    return cv2.convexHull(pts)


def hull_to_mask(img_shape: tuple[int, int, int], hull: np.ndarray) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask


def _compute_rotation(
    pts: np.ndarray, threshold: float
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    if len(pts) < 3:
        return 0.0, (0.0, 0.0), (0, 0)

    rect = cv2.minAreaRect(pts)
    center, (w, h), angle = rect

    if angle < -45:
        angle += 90
        w, h = h, w

    if abs(angle) <= threshold:
        angle = 0.0

    return angle, center, (int(round(w)), int(round(h)))


def merge_detections(
    img: np.ndarray,
    blocks: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    config = get_config()
    inpaint_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    merged_blocks = []
    for block in blocks:
        hull = block_shape_hull(block)
        if hull is not None:
            block_mask = hull_to_mask(img.shape, hull)
            inpaint_mask |= block_mask

            pts = get_block_points(block)
            angle, center, rect_size = _compute_rotation(
                pts, config.text_angle_threshold
            )

            merged_blocks.append(
                {
                    "mask": block_mask,
                    "text": block["translated_text"],
                    "angle": angle,
                    "rect_center": center,
                    "rect_size": rect_size,
                }
            )
    return inpaint_mask, merged_blocks
