import cv2
import numpy as np

from src.config import get_config, load_config
from src.debug import (
    debug_detection,
    debug_inpaint,
    debug_insertion,
    debug_merge,
    debug_translation,
)
from src.detection.merge.merge import merge_detections
from src.detection.ocr import TextDetector
from src.inpainting.lama import inpaint_page
from src.insertion.render import insert_text
from src.languages import SUPPORTED_LANGUAGES
from src.translation.llm import translate_page


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

    blocks = translate_page(img, detections, llm_source_lang, target_lang)
    if config.debug_translation:
        debug_translation(blocks)

    inpaint_mask, merged_blocks = merge_detections(img, detections, blocks)
    if config.debug_merge:
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

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        _run_pipeline(detector, img, llm_source_lang, target_lang)
