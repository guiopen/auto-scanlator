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
    block: dict, pts: np.ndarray, threshold: float
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    if len(pts) < 3:
        return 0.0, (0.0, 0.0), (0, 0)

    polygons = block.get("poly_points", [])
    angles = []
    weights = []
    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.float32)
        if len(poly_pts) >= 3:
            _, (pw, ph), p_angle = normalize_rect(poly_pts)
            angles.append(p_angle)
            weights.append(max(pw, ph))

    if angles and sum(weights) > 0:
        angle = float(np.average(angles, weights=weights))
    else:
        _, _, angle = normalize_rect(pts)

    if abs(angle) <= threshold:
        return 0.0, (0.0, 0.0), (0, 0)

    x, y, bw, bh = cv2.boundingRect(pts)
    center = (x + bw / 2.0, y + bh / 2.0)

    angle_rad = np.deg2rad(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    shifted = pts.astype(np.float32) - np.array(center, dtype=np.float32)
    rx = shifted[:, 0] * cos_a + shifted[:, 1] * sin_a
    ry = -shifted[:, 0] * sin_a + shifted[:, 1] * cos_a

    rw = int(np.ceil(2.0 * np.max(np.abs(rx)))) + 4
    rh = int(np.ceil(2.0 * np.max(np.abs(ry)))) + 4

    return angle, center, (rw, rh)


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
        assert pts is not None
        angle, center, rect_size = _compute_rotation(
            block, pts, config.text_angle_threshold
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
    return merged_blocks
