import argparse
from pathlib import Path

from PIL import Image

from pipeline import run_pipeline


def resolve_images(path: Path) -> list[Path]:
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
    args = parser.parse_args()

    image_paths = resolve_images(Path(args.image_path))
    run_pipeline([str(p) for p in image_paths])
