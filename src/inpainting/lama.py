import cv2
import numpy as np


class PageInpainter:
    def __init__(self, device: str = "cpu"):
        import torch
        from simple_lama_inpainting import SimpleLama  # type: ignore[import-untyped]

        target_device = torch.device(device)
        # simple-lama-inpainting calls torch.jit.load without map_location, which
        # restores the model on the CUDA device it was saved from
        _jit_load = torch.jit.load

        def _load(path, *args, **kwargs):
            kwargs.setdefault("map_location", target_device)
            return _jit_load(path, *args, **kwargs)

        torch.jit.load = _load
        try:
            self._lama = SimpleLama(device=target_device)
        finally:
            torch.jit.load = _jit_load

    def _build_mask(self, shape: tuple[int, ...], blocks: list[dict]) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for block in blocks:
            for poly in block.get("poly_points", []):
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return np.asarray(cv2.dilate(mask, kernel, iterations=1), dtype=np.uint8)

    def inpaint(self, img: np.ndarray, blocks: list[dict]) -> np.ndarray | None:
        mask = self._build_mask(img.shape, blocks)
        if mask.sum() == 0:
            return None

        mask = self._dilate(mask)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inpainted_pil = self._lama(rgb, mask)
        result_rgb = np.asarray(inpainted_pil)
        h, w = img.shape[:2]
        result_rgb = result_rgb[:h, :w]
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    def restore(
        self,
        cleaned: np.ndarray,
        original: np.ndarray,
        erased_blocks: list[dict],
        kept_blocks: list[dict],
    ) -> np.ndarray:
        erased = self._dilate(self._build_mask(cleaned.shape, erased_blocks))
        kept = self._dilate(self._build_mask(cleaned.shape, kept_blocks))
        restore_mask = erased.copy()
        restore_mask[kept > 0] = 0

        result = cleaned.copy()
        result[restore_mask > 0] = original[restore_mask > 0]
        return result
