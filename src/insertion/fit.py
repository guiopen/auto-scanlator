from pathlib import Path
import re

import numpy as np
from PIL import ImageFont

from src.config import get_config

FONT_PATH = Path(__file__).parent.parent.parent / "fonts" / "font.ttf"
FONT_BOLD_PATH = Path(__file__).parent.parent.parent / "fonts" / "font-bold.ttf"
_TAG_RE = re.compile(r"(</?b>)")


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return None


def _marked_width(
    text: str,
    font: ImageFont.FreeTypeFont,
    bold_font: ImageFont.FreeTypeFont | None,
) -> float:
    width = 0.0
    bold = False
    for segment in _TAG_RE.split(text):
        if segment == "<b>":
            bold = True
        elif segment == "</b>":
            bold = False
        elif segment:
            f = bold_font if (bold and bold_font is not None) else font
            width += f.getlength(segment)
    return width


def _horizontal_span(mask: np.ndarray, y: int, height: int) -> tuple[int, int] | None:
    row = mask[y : y + height, :]
    cols = np.where(row.any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


def _split_words(text: str) -> list[tuple[str, bool]]:
    tokens: list[tuple[str, bool]] = []
    bold = False
    for raw_word in text.split():
        parts = raw_word.split("-")
        for idx, part in enumerate(parts):
            token = part + "-" if idx < len(parts) - 1 else part
            if not token:
                continue
            was_bold = bold
            for seg in _TAG_RE.split(token):
                if seg == "<b>":
                    bold = True
                elif seg == "</b>":
                    bold = False
            if was_bold and not token.startswith("<b>"):
                token = "<b>" + token
            if bold and not token.endswith("</b>"):
                token = token + "</b>"
            tokens.append((token, idx == len(parts) - 1))
    return tokens


def _greedy_wrap(
    words: list[tuple[str, bool]],
    font: ImageFont.FreeTypeFont,
    bold_font: ImageFont.FreeTypeFont | None,
    mask: np.ndarray,
    y_start: int,
    font_height: int,
    line_step: int,
    mask_bottom: int,
) -> list[tuple[str, int, int, int, float]] | None:
    config = get_config()
    y = y_start
    lines = []
    i = 0

    while i < len(words):
        if y + font_height > mask_bottom:
            return None

        span = _horizontal_span(mask, y, font_height)
        if span is None:
            return None

        span_width = span[1] - span[0]
        margin_h = int(span_width * config.text_padding_h / 2)
        left = span[0] + margin_h
        right = span[1] - margin_h
        available = right - left

        test_line, prev_space = words[i]
        test_width = _marked_width(test_line, font, bold_font)
        if available <= 0 or test_width > available:
            y += line_step
            continue

        j = i + 1
        while j < len(words):
            word_text, space_after = words[j]
            sep = " " if prev_space else ""
            candidate = test_line + sep + word_text
            candidate_width = _marked_width(candidate, font, bold_font)
            if candidate_width <= available:
                test_line = candidate
                test_width = candidate_width
                prev_space = space_after
                j += 1
            else:
                break

        lines.append((test_line, y, left, right, test_width))
        y += line_step
        i = j

    return lines


def fit_text(
    text: str,
    mask: np.ndarray,
) -> tuple[int, list[tuple[str, int, int]]]:
    config = get_config()
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return 1, []

    hull_height = ys[-1] - ys[0]
    margin_v = int(hull_height * config.text_padding_v / 2)
    mask_top = int(ys[0]) + margin_v
    mask_bottom = int(ys[-1]) - margin_v
    max_vertical = mask_bottom - mask_top

    if max_vertical <= 0:
        return 1, []

    words = _split_words(text)
    if not words:
        return 1, []

    max_size = min(config.font_max_size, max_vertical)
    max_size = max(max_size, config.font_min_size)

    best_size = config.font_min_size
    best_lines = []

    lo, hi = config.font_min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(str(FONT_PATH), mid)
        bold_font = load_font(FONT_BOLD_PATH, mid)
        bbox = font.getbbox("Agy")
        font_height = int(bbox[3] - bbox[1])
        line_step = int(font_height * config.line_spacing)

        if font_height <= 0:
            hi = mid - 1
            continue

        lines = _greedy_wrap(
            words,
            font,
            bold_font,
            mask,
            mask_top,
            font_height,
            line_step,
            mask_bottom,
        )

        if lines is not None:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(str(FONT_PATH), best_size)
    bold_font = load_font(FONT_BOLD_PATH, best_size)
    bbox = font.getbbox("Agy")
    font_height = int(bbox[3] - bbox[1])
    line_step = int(font_height * config.line_spacing)

    if best_lines:
        first_y = best_lines[0][1]
        last_y = best_lines[-1][1]
        total_height = (last_y - first_y) + font_height
        available = mask_bottom - mask_top
        target_top = mask_top + (available - total_height) // 2
        offset = target_top - first_y - bbox[1]

        out = []
        for line_text, y, left, right, width in best_lines:
            line_w = _marked_width(line_text, font, bold_font)
            x = left + (right - left - line_w) / 2
            out.append((line_text, int(x), int(y + offset)))
        return best_size, out

    out = []
    y = mask_top
    for word, _ in words:
        if y + font_height > mask_bottom:
            break
        span = _horizontal_span(mask, y, font_height)
        if span is None:
            break
        span_width = span[1] - span[0]
        margin_h = int(span_width * config.text_padding_h / 2)
        left = span[0] + margin_h
        right = span[1] - margin_h
        word_w = _marked_width(word, font, bold_font)
        x = left + (right - left - word_w) / 2
        out.append((word, int(x), int(y - bbox[1])))
        y += line_step

    return best_size, out
