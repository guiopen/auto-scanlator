import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.insertion.fit import FONT_PATH, fit_text


def _render_rotated_block(
    block: dict,
    img_w: int,
    img_h: int,
) -> Image.Image | None:
    text = block["text"]
    mask = block["mask"]
    angle = block["angle"]
    center = block["rect_center"]
    rw, rh = block["rect_size"]

    if rw <= 0 or rh <= 0:
        return None

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(
        mask, rot_mat, (img_w, img_h), flags=cv2.INTER_NEAREST
    )

    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, cx - rw // 2)
    y1 = max(0, cy - rh // 2)
    x2 = min(rotated_mask.shape[1], x1 + rw)
    y2 = min(rotated_mask.shape[0], y1 + rh)
    if x2 <= x1 or y2 <= y1:
        return None
    aligned_mask = rotated_mask[y1:y2, x1:x2]

    size, lines = fit_text(text, aligned_mask)
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
    img_w, img_h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    for block in merged_blocks:
        text = block["text"]
        if not text:
            continue

        angle = block.get("angle", 0.0)

        if angle == 0.0:
            mask = block["mask"]
            size, lines = fit_text(text, mask)
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
        else:
            rotated = _render_rotated_block(block, img_w, img_h)
            if rotated is None:
                continue
            center = block["rect_center"]
            paste_x = int(round(center[0] - rotated.width / 2))
            paste_y = int(round(center[1] - rotated.height / 2))
            pil_img.paste(rotated, (paste_x, paste_y), rotated)

    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
