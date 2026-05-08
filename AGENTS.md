# AGENTS.md

## Commands
- Install deps: `uv sync`
- Run: `uv run main.py <image_or_dir> --source-lang <code> --target-lang <lang>`

## Setup requirements
- Python 3.12 (`.python-version`)
- An LLM server at `http://127.0.0.1:8080` (configurable via `config.json` but that file is denied)
- A true-type font at `fonts/font.ttf` (gitignored; required for text insertion)

## Architecture
- Entry: `main.py` → `src/cli.py` (argparse) → `src/pipeline.py`
- Pipeline stages: `detection/ocr.py` → `translation/llm.py` → `detection/merge/merge.py` → `inpainting/lama.py` → `insertion/render.py`
- Source language codes (first CLI arg) must match PaddleOCR codes listed in `src/languages.py`

## Config
- `config.json` is gitignored and denied from read access — do not attempt to read it
- All config keys have defaults in `src/config.py:Config` dataclass; use those as reference

## Important quirks
- `torch` and `simple_lama_inpainting` are lazy-imported inside `inpainting/lama.py:inpaint()`, not at module level
- Debug flags (`debug_*` in config) open blocking OpenCV windows — the program waits for a keypress per page
- Input images are resolved via `PIL.Image.open()` — any format Pillow can open is accepted
- `insertion/render.py:_render_rotated_block` uses `getRotationMatrix2D(center, -angle, …)` (straighten) and `canvas.rotate(angle, …)` (restore tilt). Do not swap signs — OpenCV angles are negative for clockwise rotation.
