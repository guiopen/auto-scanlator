import cv2
import numpy as np


def debug_merge(img: np.ndarray, mask: np.ndarray, height: int = 720):
    overlay = img.copy()
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:] = (0, 0, 255)
    overlay = np.where(mask[..., None] > 0, mask_colored, overlay)
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    h, w = result.shape[:2]
    scale = height / h
    result = cv2.resize(result, (int(w * scale), height))

    cv2.imshow("Merge", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
