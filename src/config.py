import json
from dataclasses import dataclass, fields
from pathlib import Path

_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    min_score: float = 0.5


def get() -> Config:
    try:
        data = json.loads(_CONFIG_PATH.read_text())
    except FileNotFoundError:
        return Config()

    config = Config()
    for f in fields(Config):
        if f.name in data:
            setattr(config, f.name, data[f.name])
    return config
