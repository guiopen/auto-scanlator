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
        if int(stats[i, cv2.CC_STAT_AREA]) < config.bold_char_min_area:
            continue
        h_i = float(stats[i, cv2.CC_STAT_HEIGHT])
        if h_i < config.bold_char_min_height_ratio * max_h:
            continue
        heights.append(h_i)
    return heights


def _word_score(rotated: np.ndarray, x1: int, x2: int) -> float | None:
    config = get_config()
    fg = rotated[:, x1:x2] > 0
    if not fg.any():
        return None
    dist_full = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    skel = _skeleton(fg)
    if not skel.any():
        return None
    return float((dist_full[skel] ** config.bold_depth_power).mean())


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


def _process_line(img: np.ndarray, det: dict, bg_img: np.ndarray) -> dict | None:
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

    gap_px = max(1, int(round(config.bold_word_gap_ratio * text_height)))
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
        raw = _word_score(rotated, rx1, rx2)
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
        out.append({"word": label, "quad": quad, "bold_score_raw": raw})

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


def _mark_texts(detections: list[dict], bold_words: list[dict]) -> dict[int, str]:
    bold_by_det: dict[int, list[str]] = {}
    for w in bold_words:
        if w["is_bold"]:
            bold_by_det.setdefault(w["det_idx"], []).append(w["word"])

    marked: dict[int, str] = {}
    for det_idx, labels in bold_by_det.items():
        tokens = detections[det_idx]["text"].split()
        if not tokens:
            continue
        it = iter(labels)
        next_label = next(it, None)
        out = []
        for token in tokens:
            if next_label is not None and token == next_label:
                out.append(f"<b>{token}</b>")
                next_label = next(it, None)
            else:
                out.append(token)
        marked[det_idx] = " ".join(out)
    return marked


def classify_bold(
    img: np.ndarray,
    detections: list[dict],
    blocks: list[dict],
    bg_img: np.ndarray,
) -> tuple[list[dict], float]:
    config = get_config()
    lookup = _block_lookup(blocks)
    lines = []
    for det_idx, det in enumerate(detections):
        info = _process_line(img, det, bg_img)
        if info is not None:
            lines.append((det_idx, info))

    block_heights: dict[int, list[float]] = {}
    for _, info in lines:
        block_idx = lookup.get(_poly_key(info["poly"]))
        if block_idx is not None:
            block_heights.setdefault(block_idx, []).extend(info["char_heights"])
    block_h = {
        b: (float(np.mean(hs)) if hs else 0.0) for b, hs in block_heights.items()
    }

    power = config.bold_depth_power
    bold_words = []
    for det_idx, info in lines:
        block_idx = lookup.get(_poly_key(info["poly"]))
        h_b = block_h.get(block_idx, 0.0) if block_idx is not None else 0.0
        for m in info["runs"]:
            raw = m["bold_score_raw"]
            bold_score = None if raw is None or h_b <= 0 else raw / (h_b**power)
            bold_words.append(
                {
                    "word": m["word"],
                    "quad": m["quad"],
                    "bold_score": bold_score,
                    "det_idx": det_idx,
                    "block_idx": block_idx,
                }
            )

    baseline_values = [
        w["bold_score"]
        for w in bold_words
        if w["bold_score"] is not None and _letter_count(w["word"]) > 1
    ]
    baseline = (
        float(np.percentile(baseline_values, config.bold_baseline_percentile))
        if baseline_values
        else 0.0
    )

    for w in bold_words:
        score = w["bold_score"]
        if score is None:
            w["is_bold"] = None
            continue
        n = _letter_count(w["word"])
        if n == 1:
            factor = config.bold_factor_single
        elif n == 2:
            factor = config.bold_factor_double
        else:
            factor = config.bold_factor
        w["is_bold"] = score > baseline * factor

    marked = _mark_texts(detections, bold_words)
    block_dets: dict[int, list[int]] = {}
    for det_idx, det in enumerate(detections):
        block_idx = lookup.get(_poly_key(det["poly"]))
        if block_idx is not None:
            block_dets.setdefault(block_idx, []).append(det_idx)
    for block_idx, det_indices in block_dets.items():
        texts = [marked.get(d, detections[d]["text"]) for d in det_indices]
        blocks[block_idx]["marked_text"] = " ".join(texts)

    return bold_words, baseline
