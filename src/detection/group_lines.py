import cv2
import numpy as np

from src.config import get_config


def _normalize_rect(pts: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    rect = cv2.minAreaRect(pts)
    center, (w, h), angle = rect
    if angle < -45:
        angle += 90
        w, h = h, w
    return center, (w, h), angle


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    px, py = _find(parent, x), _find(parent, y)
    if px != py:
        parent[px] = py


def group_detections(
    img: np.ndarray,
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> list[dict]:
    config = get_config()
    n = len(detections)
    if n == 0:
        return []

    rects = []
    for _, poly in detections:
        pts = np.array(poly, dtype=np.float32)
        rects.append(_normalize_rect(pts))

    expanded = []
    for center, (w, h), angle in rects:
        expanded.append((center, (w, h * 2.0), angle))

    parent = list(range(n))
    angle_threshold = config.text_angle_threshold

    for i in range(n):
        for j in range(i + 1, n):
            if abs(rects[i][2] - rects[j][2]) > angle_threshold:
                continue
            ret = cv2.rotatedRectangleIntersection(expanded[i], expanded[j])
            if ret[0] == cv2.INTERSECT_NONE:
                continue
            (cx_i, cy_i), (w_i, _), angle_i = rects[i]
            (cx_j, cy_j), (w_j, _), _ = rects[j]
            angle_rad = np.deg2rad(angle_i)
            proj_i = cx_i * np.cos(angle_rad) + cy_i * np.sin(angle_rad)
            proj_j = cx_j * np.cos(angle_rad) + cy_j * np.sin(angle_rad)
            a_i, b_i = proj_i - w_i / 2, proj_i + w_i / 2
            a_j, b_j = proj_j - w_j / 2, proj_j + w_j / 2
            overlap = max(0.0, min(b_i, b_j) - max(a_i, a_j))
            if overlap < config.merge_overlap_ratio * min(w_i, w_j):
                continue
            _union(parent, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        groups.setdefault(root, []).append(i)

    blocks = []
    for indices in groups.values():
        indices.sort()
        blocks.append({"lines": indices, "translated_text": ""})

    return blocks
