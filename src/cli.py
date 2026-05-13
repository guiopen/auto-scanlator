import argparse
from pathlib import Path

from src.images import resolve_images
from src.languages import SUPPORTED_LANGUAGES
from src.pipeline import process_pages


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

    target_lang = args.target_lang.lower()
    if target_lang not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        parser.error(
            f"Unsupported target language: '{args.target_lang}'. "
            f"Supported codes: {supported}"
        )

    image_paths = resolve_images(Path(args.image_path))
    process_pages([str(p) for p in image_paths], source_lang, target_lang)
