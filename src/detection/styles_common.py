from collections.abc import Callable

import cv2
import numpy as np
from skimage.morphology import medial_axis

from src.config import get_config
from src.utils import normalize_rect


def _binarize(crop: np.ndarray, bg_crop: np.ndarray) -> np.ndarray:
    crop_f = crop.astype(np.float32)
    dist = np.linalg.norm(crop_f - bg_crop.astype(np.float32), axis=2)
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _skeleton(binary: np.ndarray) -> np.ndarray:
    return medial_axis(binary > 0)


def _char_heights(rotated: np.ndarray) -> list[float]:
    config = get_config()
    n, _, stats, _ = cv2.connectedComponentsWithStats(rotated, connectivity=8)
    max_h = max(
        (float(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n)),
        default=0.0,
    )
    heights = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < config.char_min_area:
            continue
        h_i = float(stats[i, cv2.CC_STAT_HEIGHT])
        if h_i < config.char_min_height_ratio * max_h:
            continue
        heights.append(h_i)
    return heights


def _label_runs(line_text: str, runs: list[tuple[int, int]]) -> list[str]:
    tokens = line_text.split()
    if len(tokens) == len(runs):
        return tokens
    if not tokens:
        return [""] * len(runs)
    weights = [len(t) for t in tokens]
    boundaries = np.cumsum([w / sum(weights) for w in weights])
    span_total = float(runs[-1][1] - runs[0][0]) or 1.0
    labels = []
    for x1, x2 in runs:
        pos = ((x1 + x2) / 2.0 - runs[0][0]) / span_total
        idx = int(np.clip(np.searchsorted(boundaries, pos), 0, len(tokens) - 1))
        labels.append(tokens[idx])
    return labels


def _process_line(
    img: np.ndarray,
    det: dict,
    bg_img: np.ndarray,
    score_fn: Callable[[np.ndarray, int, int], float | None],
) -> dict | None:
    config = get_config()
    pts = np.array(det["poly"], dtype=np.float32)
    x1 = max(0, int(np.min(pts[:, 0])))
    y1 = max(0, int(np.min(pts[:, 1])))
    x2 = min(img.shape[1], int(np.max(pts[:, 0])))
    y2 = min(img.shape[0], int(np.max(pts[:, 1])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = img[y1:y2, x1:x2]
    binary = _binarize(crop, bg_img[y1:y2, x1:x2])

    center, (_, height), angle = normalize_rect(pts)
    if height <= 0:
        return None
    center_crop = (center[0] - x1, center[1] - y1)
    rot_mat = cv2.getRotationMatrix2D(center_crop, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, rot_mat, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST
    )
    inv = cv2.invertAffineTransform(rot_mat)

    rows = np.where(rotated.any(axis=1))[0]
    text_height = float(rows[-1] - rows[0] + 1) if len(rows) else height

    gap_px = max(1, int(round(config.word_gap_ratio * text_height)))
    cols = np.where(rotated.any(axis=0))[0]
    if len(cols) == 0:
        return None

    runs = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > gap_px:
            runs.append((start, prev + 1))
            start = c
        prev = c
    runs.append((start, int(cols[-1]) + 1))

    labels = _label_runs(det["text"], runs)
    out = []
    for (rx1, rx2), label in zip(runs, labels):
        raw = score_fn(rotated, rx1, rx2)
        corners = [
            (rx1, 0),
            (rx2, 0),
            (rx2, rotated.shape[0]),
            (rx1, rotated.shape[0]),
        ]
        quad = tuple(
            tuple(
                int(round(v))
                for v in (inv @ np.array([px, py, 1.0], dtype=np.float32))[:2]
                + np.array([x1, y1], dtype=np.float32)
            )
            for px, py in corners
        )
        out.append({"word": label, "quad": quad, "score_raw": raw})

    return {
        "poly": det["poly"],
        "runs": out,
        "char_heights": _char_heights(rotated),
    }


def _poly_key(poly: list) -> tuple[tuple[int, int], ...]:
    return tuple((int(pt[0]), int(pt[1])) for pt in poly)


def _letter_count(word: str) -> int:
    return sum(c.isalpha() for c in word)


def _block_lookup(blocks: list[dict]) -> dict[tuple[tuple[int, int], ...], int]:
    lookup: dict[tuple[tuple[int, int], ...], int] = {}
    for b, block in enumerate(blocks):
        for pts in block["poly_points"]:
            lookup[_poly_key(pts)] = b
    return lookup


def _mark_texts(
    detections: list[dict],
    entries: list[dict],
    tag: str,
) -> dict[int, str]:
    styled_by_det: dict[int, list[str]] = {}
    for e in entries:
        if e["styled"]:
            styled_by_det.setdefault(e["det_idx"], []).append(e["word"])

    marked: dict[int, str] = {}
    for det_idx, labels in styled_by_det.items():
        tokens = detections[det_idx]["text"].split()
        if not tokens:
            continue
        it = iter(labels)
        next_label = next(it, None)
        out = []
        for token in tokens:
            if next_label is not None and token == next_label:
                out.append(f"<{tag}>{token}</{tag}>")
                next_label = next(it, None)
            else:
                out.append(token)
        marked[det_idx] = " ".join(out)
    return marked
