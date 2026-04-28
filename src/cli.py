import argparse
from pathlib import Path

from PIL import Image

from pipeline import run_comic_translation
from utils import SUPPORTED_LANGUAGES


def _resolve_images(path: Path) -> list[Path]:
    paths = sorted(path.iterdir()) if path.is_dir() else [path]
    result = []
    for p in paths:
        if p.is_file():
            try:
                with Image.open(p):
                    result.append(p)
            except (OSError, ValueError):
                pass

    if not result:
        raise ValueError(f"No valid images found in: {path}")

    return result


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument(
        "--source-lang",
        required=True,
        help="Source language code supported by PaddleOCR",
    )
    parser.add_argument(
        "--target-lang",
        required=True,
        help="Target language for translation",
    )
    args = parser.parse_args()

    source_lang = args.source_lang.lower()
    if source_lang not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        parser.error(
            f"Unsupported source language: '{args.source_lang}'. "
            f"Supported codes: {supported}"
        )

    image_paths = _resolve_images(Path(args.image_path))
    run_comic_translation([str(p) for p in image_paths], source_lang, args.target_lang)
