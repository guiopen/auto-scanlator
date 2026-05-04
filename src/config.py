import json
from dataclasses import dataclass, field, fields
from pathlib import Path

_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    pixel_thresh: float = 0.3
    box_thresh: float = 0.6
    rec_thresh: float = 0.3
    debug_detection: bool = False
    debug_translation: bool = False
    debug_merge: bool = False
    debug_inpaint: bool = False
    debug_insertion: bool = False
    llm_api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_extra_parameters: dict = field(default_factory=dict)
    font_min_size: int = 1
    font_max_size: int = 999
    text_padding_h: int = 0
    text_padding_v: int = 0
    line_spacing: float = 1.5


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
