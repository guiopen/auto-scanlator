import cv2
import numpy as np


class PageInpainter:
    def __init__(self, device: str = "cpu"):
        import torch
        from simple_lama_inpainting import SimpleLama

        self._lama = SimpleLama(device=torch.device(device))

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if mask.sum() == 0:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inpainted_pil = self._lama(rgb, mask)
        result_rgb = np.asarray(inpainted_pil)
        h, w = img.shape[:2]
        result_rgb = result_rgb[:h, :w]
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
