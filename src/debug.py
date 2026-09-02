import json
import re

import cv2
import numpy as np


def _show_debug(title: str, img: np.ndarray, height: int = 720):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * height / h), height))
    cv2.imshow(title, resized)
    while True:
        key = cv2.waitKey(100)
        if key != -1 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def debug_detection(img: np.ndarray, detections: list, height: int = 720):
    overlay = img.copy()
    for det in detections:
        pts = np.array([[pt[0], pt[1]] for pt in det["poly"]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Detection", result, height)


def debug_bold(
    img: np.ndarray,
    bold_words: list[dict],
    baseline: float,
    height: int = 720,
):
    overlay = img.copy()
    for w in bold_words:
        pts = np.array(w["quad"], dtype=np.int32)
        if w["is_bold"] is None:
            color = (150, 150, 150)
        elif w["is_bold"]:
            color = (0, 0, 255)
        else:
            color = (0, 180, 0)
        cv2.polylines(overlay, [pts], True, color, 2)
    print(f"baseline bold_score: {baseline:.4f}")
    for w in bold_words:
        if w["is_bold"] is None:
            verdict = "unknown"
        elif w["is_bold"]:
            verdict = "BOLD"
        else:
            verdict = "normal"
        score = "n/a" if w["bold_score"] is None else f"{w['bold_score']:.4f}"
        print(f"{verdict:7s} {score:>8s}  {w['word']}")
    _show_debug("Bold", overlay, height)


def debug_translation(blocks: list[dict]):
    text = json.dumps(blocks, ensure_ascii=False, indent=2)
    text = re.sub(
        r"\[\n( *-?\d+,\n)* *-?\d+\n *\]",
        lambda m: (
            "["
            + ", ".join(x.strip().rstrip(",") for x in m.group(0).split("\n")[1:-1])
            + "]"
        ),
        text,
    )
    print(text)


def debug_grouping(
    img: np.ndarray,
    blocks: list[dict],
    height: int = 720,
):
    overlay = img.copy()
    n = len(blocks)
    for i, block in enumerate(blocks):
        hue = int((180 * i) / max(n, 1))
        color = cv2.cvtColor(
            np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        for poly in block.get("poly_points", []):
            pts = np.array(poly, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Grouping", result, height)


def debug_merge(
    img: np.ndarray,
    merged_blocks: list[dict],
    height: int = 720,
):
    overlay = img.copy()
    n = len(merged_blocks)
    for i, block in enumerate(merged_blocks):
        hue = int((180 * i) / max(n, 1))
        color = cv2.cvtColor(
            np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        mask = block.get("mask")
        if mask is not None:
            overlay[mask > 0] = color
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Merge", result, height)


def debug_inpaint(inpainted: np.ndarray, height: int = 720):
    _show_debug("Inpaint", inpainted, height)


def debug_insertion(before: np.ndarray, after: np.ndarray, height: int = 720):
    side_by_side = np.hstack([before, after])
    _show_debug("Insertion (before | after)", side_by_side, height)
