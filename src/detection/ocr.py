from paddleocr import PaddleOCR

import config

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def detect(image_path: str) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    cfg = config.get()
    result = ocr.predict(image_path)
    detections = []
    for res in result:
        for text, score, poly in zip(
            res["rec_texts"], res["rec_scores"], res["rec_polys"]
        ):
            if text.strip() and score >= cfg.min_score:
                detections.append(
                    (text, tuple(tuple(int(v) for v in pt) for pt in poly))
                )
    return detections
