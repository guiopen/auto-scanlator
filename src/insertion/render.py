import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.insertion.fit import FONT_PATH, fit_text


def _render_rotated_block(block: dict) -> Image.Image | None:
    text = block["text"]
    angle = block["angle"]
    rw, rh = block["rect_size"]

    if rw <= 0 or rh <= 0:
        return None

    size, lines = fit_text(text, rw, rh)
    if not lines:
        return None

    font = ImageFont.truetype(str(FONT_PATH), size)
    canvas = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for line_text, x, y in lines:
        draw.text(
            (x, y),
            line_text,
            font=font,
            fill="black",
            stroke_width=2,
            stroke_fill="white",
        )

    rotated_canvas = canvas.rotate(-angle, expand=True)
    return rotated_canvas


def insert_text(
    img: np.ndarray,
    merged_blocks: list[dict],
) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    for block in merged_blocks:
        text = block["text"]
        if not text:
            continue

        angle = block.get("angle", 0.0)
        center = block["rect_center"]
        rw, rh = block["rect_size"]

        if rw <= 0 or rh <= 0:
            continue

        if angle == 0.0:
            cx, cy = int(round(center[0])), int(round(center[1]))
            x1 = cx - rw // 2
            y1 = cy - rh // 2

            size, lines = fit_text(text, rw, rh)
            font = ImageFont.truetype(str(FONT_PATH), size)
            for line_text, x, y in lines:
                draw.text(
                    (x1 + x, y1 + y),
                    line_text,
                    font=font,
                    fill="black",
                    stroke_width=2,
                    stroke_fill="white",
                )
        else:
            rotated = _render_rotated_block(block)
            if rotated is None:
                continue
            paste_x = int(round(center[0] - rotated.width / 2))
            paste_y = int(round(center[1] - rotated.height / 2))
            pil_img.paste(rotated, (paste_x, paste_y), rotated)

    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
