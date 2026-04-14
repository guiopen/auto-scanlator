from paddleocr import PaddleOCR

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr


def detect(image_path: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    result = _get_ocr().predict(image_path)
    detections = []
    for res in result:
        for text, box in zip(res["rec_texts"], res["rec_boxes"]):
            if text.strip():
                detections.append((text, tuple(int(v) for v in box)))
    return detections
