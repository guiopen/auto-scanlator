import json
import urllib.request

_URL = "http://127.0.0.1:8080/completion"


def translate(texts: list[str], source_lang: str, target_lang: str) -> dict[str, str]:
    prompt = (
        f"Translate the following comic book speech bubble texts from {source_lang} to {target_lang}.\n"
        f"Return ONLY a JSON object mapping each original text to its translation.\n"
        f"Texts: {json.dumps(texts, ensure_ascii=False)}"
    )
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": 2048,
            "json_schema": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        }
    ).encode()
    req = urllib.request.Request(
        _URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        response = json.loads(resp.read())
        result = json.loads(response["content"])
        print(f"Translation result: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result
