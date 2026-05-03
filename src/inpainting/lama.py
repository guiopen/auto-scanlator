import cv2
import numpy as np


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    import torch
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama(device=torch.device("cpu"))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inpainted_pil = lama(rgb, mask)
    result_rgb = np.asarray(inpainted_pil)
    h, w = img.shape[:2]
    result_rgb = result_rgb[:h, :w]
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def inpaint_page(
    img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if mask.sum() == 0:
        return None

    return inpaint(img, mask)
