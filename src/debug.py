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
    for _, poly in detections:
        pts = np.array([[pt[0], pt[1]] for pt in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Detection", result, height)


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
    detections: list,
    blocks: list[dict],
    height: int = 720,
):
    overlay = img.copy()
    n = len(blocks)
    for i, block in enumerate(blocks):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(
            np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        for line_idx in block["lines"]:
            _, poly = detections[line_idx]
            pts = np.array([[pt[0], pt[1]] for pt in poly], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Grouping", result, height)


def debug_merge(img: np.ndarray, mask: np.ndarray, height: int = 720):
    overlay = img.copy()
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:] = (0, 0, 255)
    overlay = np.where(mask[..., None] > 0, mask_colored, overlay)
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    _show_debug("Merge", result, height)


def debug_inpaint(inpainted: np.ndarray, height: int = 720):
    _show_debug("Inpaint", inpainted, height)


def debug_insertion(before: np.ndarray, after: np.ndarray, height: int = 720):
    side_by_side = np.hstack([before, after])
    _show_debug("Insertion (before | after)", side_by_side, height)
