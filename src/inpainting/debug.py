import cv2
import numpy as np


def debug_inpaint(inpainted: np.ndarray, height: int = 720):
    h, w = inpainted.shape[:2]
    scale = height / h
    resized = cv2.resize(inpainted, (int(w * scale), height))
    cv2.imshow("Inpaint", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
