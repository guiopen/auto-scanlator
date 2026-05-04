import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.insertion.fit import FONT_PATH, fit_text


def insert_text(
    img: np.ndarray,
    merged_blocks: list[dict],
    cfg,
) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    for block in merged_blocks:
        text = block["text"]
        mask = block["mask"]
        if not text:
            continue

        size, lines = fit_text(text, mask, cfg)
        font = ImageFont.truetype(str(FONT_PATH), size)

        for line_text, x, y in lines:
            draw.text(
                (x, y),
                line_text,
                font=font,
                fill="black",
                stroke_width=2,
                stroke_fill="white",
            )

    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
