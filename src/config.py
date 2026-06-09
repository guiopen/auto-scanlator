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
    debug_grouping: bool = False
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
    text_padding_h: float = 0.05
    text_padding_v: float = 0.05
    line_spacing: float = 1.5
    text_angle_threshold: float = 2.0
    merge_overlap_ratio: float = 0.5


_config: Config | None = None


def load_config() -> Config:
    global _config
    try:
        data = json.loads(_CONFIG_PATH.read_text())
    except FileNotFoundError:
        _config = Config()
        return _config

    config = Config()
    for f in fields(Config):
        if f.name in data:
            setattr(config, f.name, data[f.name])
    _config = config
    return _config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
