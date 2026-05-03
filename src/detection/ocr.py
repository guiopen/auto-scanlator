import numpy as np
from paddleocr import PaddleOCR

from src.config import Config


class TextDetector:
    def __init__(self, cfg: Config, lang: str):
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=cfg.pixel_thresh,
            text_det_box_thresh=cfg.box_thresh,
            text_rec_score_thresh=cfg.rec_thresh,
        )

    def detect(self, img: np.ndarray) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
        return [
            (str(text), tuple((int(pt[0]), int(pt[1])) for pt in poly))
            for res in self._ocr.predict(img)
            for text, poly in zip(res["rec_texts"], res["rec_polys"])
            if str(text).strip()
        ]
