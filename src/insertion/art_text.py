import base64
import json
import math
import urllib.request

import cv2
import numpy as np

from src.config import get_config
from src.debug import debug_art_text
from src.insertion.merge import block_shape_hull

_ASPECT_RATIOS = (
    "1:1",
    "1:2",
    "1:4",
    "2:1",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
)
_MODEL_SIZE = 1024


def _ratio_value(label: str) -> float:
    width, height = label.split(":")
    return int(width) / int(height)


def _fit_aspect(width: int, height: int) -> tuple[str, int, int]:
    ratio = width / height
    label = min(
        _ASPECT_RATIOS,
        key=lambda r: abs(math.log(_ratio_value(r) / ratio)),
    )
    target = _ratio_value(label)
    if ratio < target:
        return label, max(width, round(height * target)), height
    return label, width, max(height, round(width / target))


def _block_rect(block: dict) -> tuple[int, int, int, int] | None:
    hull = block_shape_hull(block)
    if hull is None:
        return None
    x, y, w, h = cv2.boundingRect(hull)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    crop = img[y1:y2, x1:x2]
    borders = (y1 - y, (y + h) - y2, x1 - x, (x + w) - x2)
    if any(borders):
        crop = cv2.copyMakeBorder(crop, *borders, cv2.BORDER_REPLICATE)
    return crop


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    shrinks = width * height < img.shape[0] * img.shape[1]
    interpolation = cv2.INTER_AREA if shrinks else cv2.INTER_LANCZOS4
    return cv2.resize(img, (width, height), interpolation=interpolation)


def _for_model(crop: np.ndarray) -> np.ndarray:
    crop_h, crop_w = crop.shape[:2]
    scale = _MODEL_SIZE / max(crop_h, crop_w)
    if abs(scale - 1.0) < 1e-3:
        return crop
    return _resize(crop, max(1, round(crop_w * scale)), max(1, round(crop_h * scale)))


def _plain(text: str) -> str:
    return text.replace("<b>", "").replace("</b>", "")


def _build_prompt(translated: str) -> str:
    return (
        f'Replace the text in the image with "{translated}", '
        "preserving the font, color, angle, perspective and style of the "
        "original lettering. Keep the rest of the image unchanged."
    )


def _encode(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _call_api(crop: np.ndarray, prompt: str, aspect_ratio: str) -> np.ndarray | None:
    config = get_config()
    payload = {
        "model": config.art_text_model,
        "prompt": prompt,
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode(crop)}"},
            }
        ],
        "resolution": config.art_text_resolution,
        "aspect_ratio": aspect_ratio,
        "n": config.art_text_n,
        "seed": config.art_text_seed,
    }

    headers = {"Content-Type": "application/json"}
    if config.art_text_api_key:
        headers["Authorization"] = f"Bearer {config.art_text_api_key}"

    req = urllib.request.Request(
        config.art_text_api_url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )

    with urllib.request.urlopen(req) as resp:
        raw = json.loads(resp.read())

    for item in raw.get("data", []):
        b64 = item.get("b64_json")
        if not b64:
            continue
        buffer = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return None


def _paste(page: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    img_h, img_w = page.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2 = min(img_w, x + patch.shape[1])
    y2 = min(img_h, y + patch.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    page[y1:y2, x1:x2] = patch[y1 - y : y2 - y, x1 - x : x2 - x]


def replace_art_text(page: np.ndarray, art_blocks: list[dict]) -> np.ndarray:
    config = get_config()
    result = page.copy()
    for index, block in enumerate(art_blocks):
        rect = _block_rect(block)
        if rect is None:
            continue

        x, y, w, h = rect
        aspect_ratio, crop_w, crop_h = _fit_aspect(w, h)
        crop_x = x + w // 2 - crop_w // 2
        crop_y = y + h // 2 - crop_h // 2

        crop = _for_model(_crop(result, crop_x, crop_y, crop_w, crop_h))
        prompt = _build_prompt(_plain(block["translated_text"]))
        generated = _call_api(crop, prompt, aspect_ratio)
        if generated is None:
            continue

        if config.debug_art_text:
            debug_art_text(index, crop, generated)

        _paste(result, _resize(generated, crop_w, crop_h), crop_x, crop_y)
    return result
