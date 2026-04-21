import json
from dataclasses import dataclass, fields
from pathlib import Path

_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    pixel_thresh: float = 0.3
    box_thresh: float = 0.6
    rec_thresh: float = 0.3
    debug_detection: bool = False


def get_config() -> Config:
    try:
        data = json.loads(_CONFIG_PATH.read_text())
    except FileNotFoundError:
        return Config()

    config = Config()
    for f in fields(Config):
        if f.name in data:
            setattr(config, f.name, data[f.name])
    return config
