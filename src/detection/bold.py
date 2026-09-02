import cv2
import numpy as np

from src.config import get_config
from src.detection.styles_common import (
    _block_lookup,
    _letter_count,
    _mark_texts,
    _poly_key,
    _process_line,
    _skeleton,
)


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
        info = _process_line(img, det, bg_img, _word_score)
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
            raw = m["score_raw"]
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

    entries = [
        {"word": w["word"], "det_idx": w["det_idx"], "styled": w["is_bold"] is True}
        for w in bold_words
    ]
    marked = _mark_texts(detections, entries, "b")
    block_dets: dict[int, list[int]] = {}
    for det_idx, det in enumerate(detections):
        block_idx = lookup.get(_poly_key(det["poly"]))
        if block_idx is not None:
            block_dets.setdefault(block_idx, []).append(det_idx)
    for block_idx, det_indices in block_dets.items():
        texts = [marked.get(d, detections[d]["text"]) for d in det_indices]
        blocks[block_idx]["marked_text"] = " ".join(texts)

    return bold_words, baseline
