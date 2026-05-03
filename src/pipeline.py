import cv2
import numpy as np

from src.config import get_config
from src.detection.debug import debug_detection
from src.detection.merge.debug import debug_merge
from src.detection.merge.merge import merge_detections
from src.detection.ocr import TextDetector
from src.inpainting.debug import debug_inpaint
from src.inpainting.lama import inpaint_page
from src.languages import SUPPORTED_LANGUAGES
from src.translation.debug import debug_translation
from src.translation.llm import translate_page


def _run_pipeline(
    cfg,
    detector: TextDetector,
    img: np.ndarray,
    llm_source_lang: str,
    target_lang: str,
):
    detections = detector.detect(img)
    if cfg.debug_detection:
        debug_detection(img, detections)

    blocks = translate_page(
        img,
        detections,
        cfg.llm_api_url,
        llm_source_lang,
        target_lang,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        extra_parameters=cfg.llm_extra_parameters,
    )
    if cfg.debug_translation:
        debug_translation(blocks)

    mask = merge_detections(img, detections, blocks)
    if cfg.debug_merge:
        debug_merge(img, mask)

    result = inpaint_page(img, mask)
    if cfg.debug_inpaint and result is not None:
        debug_inpaint(result)


def process_pages(image_paths: list[str], source_lang: str, target_lang: str):
    cfg = get_config()
    detector = TextDetector(cfg, source_lang)

    llm_source_lang = SUPPORTED_LANGUAGES[source_lang]

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        _run_pipeline(cfg, detector, img, llm_source_lang, target_lang)
