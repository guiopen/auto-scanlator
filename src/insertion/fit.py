from pathlib import Path

from PIL import ImageFont

FONT_PATH = Path(__file__).parent.parent.parent / "fonts" / "font.ttf"


def _word_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        if font.getlength(test) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _layout(
    text: str, size: int, max_width: int,
) -> tuple[ImageFont.FreeTypeFont, int, list[str], float, int] | None:
    font = ImageFont.truetype(str(FONT_PATH), size)
    bbox = font.getbbox("Agy")
    font_height = int(bbox[3] - bbox[1])
    if font_height <= 0:
        return None
    lines = _word_wrap(text, font, max_width)
    max_line_width = max((font.getlength(ln) for ln in lines), default=0.0)
    total_height = int(font_height * len(lines))
    return font, font_height, lines, max_line_width, total_height


def fit_text(
    text: str,
    max_width: int,
    max_height: int,
) -> tuple[int, list[tuple[str, int, int]]]:
    if max_width <= 0 or max_height <= 0 or not text.strip():
        return 1, []

    font_size = max_height

    result = _layout(text, font_size, max_width)
    if result is None:
        return 1, []
    font, font_height, lines, max_lw, total_h = result

    scale_w = max_width / max_lw if max_lw > max_width else 1.0
    scale_h = max_height / total_h if total_h > max_height else 1.0
    scale = min(scale_w, scale_h)

    if scale < 1.0:
        font_size = max(1, int(font_size * scale))
        result = _layout(text, font_size, max_width)
        if result is None:
            return 1, []
        font, font_height, lines, max_lw, total_h = result

    offset_y = (max_height - total_h) // 2
    result = []
    y = offset_y
    for line_text in lines:
        line_w = font.getlength(line_text)
        x = (max_width - line_w) // 2
        result.append((line_text, int(x), int(y)))
        y += font_height

    return font_size, result
