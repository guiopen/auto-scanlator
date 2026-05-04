from pathlib import Path

import numpy as np
from PIL import ImageFont

FONT_PATH = Path(__file__).parent.parent.parent / "fonts" / "font.ttf"


def _horizontal_span(mask: np.ndarray, y: int, height: int) -> tuple[int, int] | None:
    row = mask[y : y + height, :]
    cols = np.where(row.any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


def _greedy_wrap(
    words: list[str],
    font: ImageFont.FreeTypeFont,
    mask: np.ndarray,
    y_start: int,
    font_height: int,
    line_step: int,
    cfg,
    mask_bottom: int,
) -> list[tuple[str, int, int, int]] | None:
    y = y_start
    lines = []
    i = 0

    while i < len(words):
        span = _horizontal_span(mask, y, font_height)
        if span is None:
            return None

        span_width = span[1] - span[0]
        margin_h = int(span_width * cfg.text_padding_h / 200)
        left = span[0] + margin_h
        right = span[1] - margin_h
        available = right - left

        if available <= 0:
            return None

        test_line = words[i]
        test_width = font.getlength(test_line)
        if test_width > available:
            return None

        j = i + 1
        while j < len(words):
            candidate = test_line + " " + words[j]
            candidate_width = font.getlength(candidate)
            if candidate_width <= available:
                test_line = candidate
                test_width = candidate_width
                j += 1
            else:
                break

        lines.append((test_line, y, left, right, test_width))
        y += line_step

        if y + font_height > mask_bottom and j < len(words):
            return None

        i = j

    return lines


def fit_text(
    text: str,
    mask: np.ndarray,
    cfg,
) -> tuple[int, list[tuple[str, int, int]]]:
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return 1, []

    hull_height = ys[-1] - ys[0]
    margin_v = int(hull_height * cfg.text_padding_v / 200)
    mask_top = int(ys[0]) + margin_v
    mask_bottom = int(ys[-1]) - margin_v
    max_vertical = mask_bottom - mask_top

    if max_vertical <= 0:
        return 1, []

    words = text.split()
    if not words:
        return 1, []

    max_size = min(cfg.font_max_size, max_vertical)
    max_size = max(max_size, cfg.font_min_size)

    best_size = cfg.font_min_size
    best_lines = []

    lo, hi = cfg.font_min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(str(FONT_PATH), mid)
        bbox = font.getbbox("Agy")
        font_height = bbox[3] - bbox[1]
        line_step = int(font_height * cfg.line_spacing)

        if font_height <= 0:
            hi = mid - 1
            continue

        lines = _greedy_wrap(
            words,
            font,
            mask,
            mask_top,
            font_height,
            line_step,
            cfg,
            mask_bottom,
        )

        if lines is not None:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(str(FONT_PATH), best_size)
    bbox = font.getbbox("Agy")
    font_height = bbox[3] - bbox[1]
    line_step = int(font_height * cfg.line_spacing)

    if best_lines:
        total_height = font_height + (len(best_lines) - 1) * line_step
        available = mask_bottom - mask_top
        offset = max(0, (available - total_height) // 2)

        out = []
        for line_text, y, left, right, width in best_lines:
            line_w = font.getlength(line_text)
            x = left + (right - left - line_w) / 2
            out.append((line_text, int(x), int(y + offset)))
        return best_size, out

    out = []
    y = mask_top
    for word in words:
        if y + font_height > mask_bottom:
            break
        span = _horizontal_span(mask, y, font_height)
        if span is None:
            break
        span_width = span[1] - span[0]
        margin_h = int(span_width * cfg.text_padding_h / 200)
        left = span[0] + margin_h
        right = span[1] - margin_h
        word_w = font.getlength(word)
        x = left + (right - left - word_w) / 2
        out.append((word, int(x), int(y)))
        y += line_step

    return best_size, out
