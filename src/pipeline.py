from config import get_config
from detection.ocr import TextDetector
from inpainting.lama import inpaint_page
from translation.llm import translate_page
from utils import SUPPORTED_LANGUAGES, debug_detection, debug_inpaint, debug_translation


def _run_pipeline(
    cfg,
    detector: TextDetector,
    image_path: str,
    llm_source_lang: str,
    target_lang: str,
):
    detections = detector.detect(image_path)
    if cfg.debug_detection:
        debug_detection(image_path, detections)

    blocks = translate_page(
        image_path,
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

    result = inpaint_page(image_path, detections, blocks)
    if cfg.debug_inpaint and result is not None:
        debug_inpaint(result)


def process_pages(image_paths: list[str], source_lang: str, target_lang: str):
    cfg = get_config()
    detector = TextDetector(cfg, source_lang)

    llm_source_lang = SUPPORTED_LANGUAGES[source_lang]

    for image_path in image_paths:
        _run_pipeline(cfg, detector, image_path, llm_source_lang, target_lang)
