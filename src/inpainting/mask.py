import cv2
import numpy as np


def get_block_points(
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
    block: dict,
) -> np.ndarray | None:
    line_indices = block.get("lines", [])
    points = []
    for idx in line_indices:
        if 0 <= idx < len(detections):
            _, poly = detections[idx]
            for pt in poly:
                points.append([pt[0], pt[1]])
    return np.array(points, dtype=np.int32) if points else None


def block_shape_hull(
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
    block: dict,
) -> np.ndarray | None:
    pts = get_block_points(detections, block)
    if pts is None or len(pts) < 3:
        return None
    return cv2.convexHull(pts)


def hull_to_mask(img_shape: tuple[int, int, int], hull: np.ndarray) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def create_inpaint_mask(
    img_shape: tuple[int, int, int],
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
    blocks: list[dict],
) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for block in blocks:
        hull = block_shape_hull(detections, block)
        if hull is not None:
            mask |= hull_to_mask(img_shape, hull)
    return mask
