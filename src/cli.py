import argparse
from pathlib import Path

from PIL import Image

from pipeline import run_pipeline


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    path = Path(args.image_path)

    if path.is_dir():
        for image_path in sorted(path.iterdir()):
            if image_path.is_file():
                try:
                    with Image.open(image_path):
                        run_pipeline(str(image_path))
                except (OSError, ValueError):
                    pass
        return

    run_pipeline(args.image_path)
