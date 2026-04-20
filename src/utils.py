import cv2
import numpy as np


def debug_detection(image_path: str, detections: list, height: int = 720):
    img = cv2.imread(image_path)
    overlay = img.copy()

    for _, poly in detections:
        pts = np.array([[pt[0], pt[1]] for pt in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))

    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    h, w = result.shape[:2]
    scale = height / h
    result = cv2.resize(result, (int(w * scale), height))

    cv2.imshow("Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
