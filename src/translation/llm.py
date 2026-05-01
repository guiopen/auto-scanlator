import base64
import json
import urllib.request
from pathlib import Path

from src.translation.prompt import _SYSTEM_PROMPT_TEMPLATE


def _encode_image(image_path: str) -> str:
    img_bytes = Path(image_path).read_bytes()
    return base64.b64encode(img_bytes).decode("ascii")


def _build_api_payload(
    model: str, b64_image: str, prompt: str, extra_parameters: dict | None
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 32768,
    }
    if extra_parameters:
        payload.update(extra_parameters)
    return payload


def _call_llm_api(api_url: str, payload: dict, api_key: str) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )

    with urllib.request.urlopen(req) as resp:
        raw = json.loads(resp.read())

    return raw.get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_llm_response(content: str) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        content = content.rsplit("```", 1)[0].strip()
    try:
        blocks = json.loads(content)
        if isinstance(blocks, list):
            return blocks
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def translate_page(
    image_path: str,
    lines: list[tuple[str, tuple[tuple[int, int], ...]]],
    api_url: str,
    source_lang: str,
    target_lang: str,
    api_key: str = "",
    model: str = "",
    extra_parameters: dict | None = None,
):
    prompt_lines = "\n".join(f"Line {i}: {text}" for i, (text, _) in enumerate(lines))
    full_prompt = (
        _SYSTEM_PROMPT_TEMPLATE.substitute(
            source_lang=source_lang, target_lang=target_lang
        )
        + "\n\nLines detected:\n"
        + prompt_lines
    )

    b64 = _encode_image(image_path)
    payload = _build_api_payload(model, b64, full_prompt, extra_parameters)
    content = _call_llm_api(api_url, payload, api_key)
    return _parse_llm_response(content)
