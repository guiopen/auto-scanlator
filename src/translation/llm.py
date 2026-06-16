import base64
import json
import urllib.request

import cv2
import numpy as np

from src.config import get_config
from src.translation.prompt import _SYSTEM_PROMPT_TEMPLATE


def _encode_image(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _build_api_payload(b64_image: str, prompt: str) -> dict:
    config = get_config()
    payload = {
        "model": config.llm_model,
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
    if config.llm_extra_parameters:
        payload.update(config.llm_extra_parameters)
    return payload


def _call_llm_api(payload: dict) -> str:
    config = get_config()
    headers = {"Content-Type": "application/json"}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"

    req = urllib.request.Request(
        config.llm_api_url,
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
    img: np.ndarray,
    grouped_blocks: list[dict],
    source_lang: str,
    target_lang: str,
) -> list[dict]:
    prompt_lines = "\n".join(
        f"Block {i}: {block['original_text']}"
        for i, block in enumerate(grouped_blocks)
    )
    full_prompt = (
        _SYSTEM_PROMPT_TEMPLATE.substitute(
            source_lang=source_lang, target_lang=target_lang
        )
        + "\n\nBlocks detected:\n"
        + prompt_lines
    )

    b64 = _encode_image(img)
    payload = _build_api_payload(b64, full_prompt)
    content = _call_llm_api(payload)

    llm_response = _parse_llm_response(content)
    indexed = {
        b["block"]: b.get("translated_text", "")
        for b in llm_response
        if isinstance(b, dict) and "block" in b
    }

    result = []
    for i, block in enumerate(grouped_blocks):
        if i in indexed:
            result.append({**block, "translated_text": indexed[i]})
    return result
