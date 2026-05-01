import cv2
import numpy as np


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    image_path: str,
    detections: list[tuple[str, tuple[tuple[int, int], ...]]],
    blocks: list[dict],
) -> np.ndarray | None:
    from src.inpainting.mask import create_inpaint_mask

    img = cv2.imread(image_path)
    if img is None:
        return None

    mask = create_inpaint_mask(img.shape, detections, blocks)
    if mask.sum() == 0:
        return None

    return inpaint(img, mask)
