import base64
import json
import urllib.request
from pathlib import Path
from string import Template

_SYSTEM_PROMPT_TEMPLATE = Template(
    "You are an expert comic translator.\n"
    "\n"
    "You receive:\n"
    "1. An image of a comic page\n"
    "2. A list of text lines detected by OCR, each with a unique ID\n"
    "\n"
    "The comic page is written in $source_lang and must be translated to $target_lang.\n"
    "\n"
    "IMPORTANT: The OCR often makes mistakes and detects text that SHOULD NOT be translated.\n"
    "This includes but is not limited to:\n"
    "- Sound effects (BOOM, POW, CRASH, BAM, ZAP, etc.)\n"
    "- Street signs, traffic signs, store signs in the background\n"
    "- Computer screens, monitors, panels with system text\n"
    "- License plates, posters, advertisements on walls\n"
    "- Chapter titles or volume numbers that are part of the page design\n"
    "- Artist names, copyright notices, publisher logos\n"
    "- Background text on objects (clothing labels, book spines, food packaging)\n"
    "- Any text that is part of the environment/scenery rather than speech or narration\n"
    "\n"
    'The OCR can also misread text: merged words, missing letters, fragmented lines, or garbled characters. Always cross-check with the image itself. Only correct OCR detection errors; never "fix" intentional author style, slang, dialect, or deliberate misspellings.\n'
    "\n"
    "Your task:\n"
    "1. Identify which lines are actual dialogue, narration, or thoughts that need translation\n"
    "2. Group lines that belong to the same speech bubble, text block, or narrative context\n"
    "3. Translate each group as a cohesive text block (not line by line)\n"
    "4. Completely IGNORE any line that is a sound effect, background text, or otherwise should not be translated\n"
    "\n"
    "CRITICAL: Do NOT return blocks for lines that should not be translated. Simply omit them entirely.\n"
    "Only return blocks for text that genuinely needs translation.\n"
    "\n"
    "Return ONLY a JSON array of blocks:\n"
    "[\n"
    "  {\n"
    '    "lines": [0, 1, 2],\n'
    '    "translated_text": "Natural translation of the entire block"\n'
    "  }\n"
    "]\n"
    "\n"
    "Rules:\n"
    "- Each line ID must appear in exactly ONE block, or be omitted if it should not be translated\n"
    "- translated_text should be natural and fluent, not literal word-by-word\n"
    "- If the original text spans multiple lines in a bubble, translate it as continuous prose\n"
    "- Never include sound effects, background signs, or environmental text\n"
    "- Even if speech bubbles are connected, treat each one as a separate block"
)


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

    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ],
        "max_tokens": 32768,
    }

    if extra_parameters:
        payload.update(extra_parameters)

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

    content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
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
