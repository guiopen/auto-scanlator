import cv2
import numpy as np

from src.config import get_config, load_config
from src.debug import (
    debug_detection,
    debug_inpaint,
    debug_insertion,
    debug_translation,
)
from src.detection.ocr import TextDetector
from src.inpainting.lama import inpaint_page
from src.insertion.render import insert_text
from src.languages import SUPPORTED_LANGUAGES
from src.translation.llm import translate_page


def _compute_rotation(
    pts: np.ndarray, threshold: float
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    if len(pts) < 3:
        return 0.0, (0.0, 0.0), (0, 0)
    rect = cv2.minAreaRect(pts)
    center = (float(rect[0][0]), float(rect[0][1]))
    (w, h), angle = rect[1], rect[2]
    if angle < -45:
        angle += 90
        w, h = h, w
    if abs(angle) <= threshold:
        angle = 0.0
    return angle, center, (int(round(w)), int(round(h)))


def _build_mask_and_blocks(
    img: np.ndarray,
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
    translated_lines: list[dict],
    angle_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    inpaint_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    merged_blocks = []

    for tl in translated_lines:
        line_idx = tl["line"]
        if line_idx < 0 or line_idx >= len(detections):
            continue
        _, poly = detections[line_idx]
        pts = np.array([[pt[0], pt[1]] for pt in poly], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(inpaint_mask, [pts], 255)
        angle, center, rect_size = _compute_rotation(pts, angle_threshold)
        merged_blocks.append(
            {
                "text": tl["translated_text"],
                "angle": angle,
                "rect_center": center,
                "rect_size": rect_size,
            }
        )

    return inpaint_mask, merged_blocks


def _run_pipeline(
    detector: TextDetector,
    img: np.ndarray,
    llm_source_lang: str,
    target_lang: str,
):
    config = get_config()
    detections = detector.detect(img)
    if config.debug_detection:
        debug_detection(img, detections)

    translated_lines = translate_page(img, detections, llm_source_lang, target_lang)
    if config.debug_translation:
        debug_translation(translated_lines)

    inpaint_mask, merged_blocks = _build_mask_and_blocks(
        img, detections, translated_lines, config.text_angle_threshold
    )
    if config.debug_merge and np.any(inpaint_mask):
        from src.debug import debug_merge
        debug_merge(img, inpaint_mask)

    cleaned_page = inpaint_page(img, inpaint_mask)
    if config.debug_inpaint and cleaned_page is not None:
        debug_inpaint(cleaned_page)

    if cleaned_page is not None:
        result = insert_text(cleaned_page, merged_blocks)
        if config.debug_insertion:
            debug_insertion(img, result)


def process_pages(image_paths: list[str], source_lang: str, target_lang: str):
    load_config()
    detector = TextDetector(SUPPORTED_LANGUAGES[source_lang].ocr_code)

    llm_source_lang = SUPPORTED_LANGUAGES[source_lang].label
    llm_target_lang = SUPPORTED_LANGUAGES[target_lang].label

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        _run_pipeline(detector, img, llm_source_lang, llm_target_lang)
