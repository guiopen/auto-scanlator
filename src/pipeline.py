from config import get_config
from detection.ocr import TextDetector
from utils import debug_detection

cfg = get_config()
detector = TextDetector(cfg)


def run_pipeline(image_paths: list[str]):
    for image_path in image_paths:
        detections = detector.detect(image_path)

        if cfg.debug_detection:
            debug_detection(image_path, detections)
