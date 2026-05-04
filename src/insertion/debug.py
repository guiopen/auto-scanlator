import cv2
import numpy as np


def debug_insertion(before: np.ndarray, after: np.ndarray, height: int = 720):
    side_by_side = np.hstack([before, after])
    h, w = side_by_side.shape[:2]
    scale = height / h
    resized = cv2.resize(side_by_side, (int(w * scale), height))
    cv2.imshow("Insertion (before | after)", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
