from detection.ocr import TextDetector
import config
from utils import debug_detection


cfg = config.get()
detector = TextDetector(cfg)


def run_pipeline(image_path: str):
    detections = detector.detect(image_path)

    if cfg.debug_detection:
        debug_detection(image_path, detections)
