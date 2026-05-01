from pathlib import Path

from PIL import Image


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
