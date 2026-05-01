import json
import re

import cv2
import numpy as np

SUPPORTED_LANGUAGES = {
    "ch": "CHINESE",
    "en": "ENGLISH",
    "fr": "FRENCH",
    "de": "GERMAN",
    "japan": "JAPANESE",
    "korean": "KOREAN",
    "chinese_cht": "CHINESE TRADITIONAL",
    "af": "AFRIKAANS",
    "it": "ITALIAN",
    "es": "SPANISH",
    "bs": "BOSNIAN",
    "pt": "PORTUGUESE",
    "cs": "CZECH",
    "cy": "WELSH",
    "da": "DANISH",
    "et": "ESTONIAN",
    "ga": "IRISH",
    "hr": "CROATIAN",
    "uz": "UZBEK",
    "ru": "RUSSIAN",
    "uk": "UKRAINIAN",
    "el": "GREEK",
    "ku": "KURDISH",
    "mt": "MALTESE",
    "ro": "ROMANIAN",
    "fi": "FINNISH",
    "gl": "GALICIAN",
    "rm": "ROMANSH",
    "ca": "CATALAN",
    "qu": "QUECHUA",
    "te": "TELUGU",
    "rs_cyrillic": "SERBIAN CYRILLIC",
    "bg": "BULGARIAN",
    "mn": "MONGOLIAN",
    "ab": "ABKHAZ",
    "ady": "ADYGHE",
    "kbd": "KABARDIAN",
    "av": "AVAR",
    "dar": "DARGWA",
    "inh": "INGUSH",
    "ce": "CHECHEN",
    "lki": "LAK",
    "lez": "LEZGIAN",
    "tab": "TABASARAN",
    "kk": "KAZAKH",
    "ky": "KYRGYZ",
    "tg": "TAJIK",
    "mk": "MACEDONIAN",
    "tt": "TATAR",
    "cv": "CHUVASH",
    "ba": "BASHKIR",
    "mhr": "MARI",
    "mo": "MOLDOVAN",
    "udm": "UDMURT",
    "kv": "KOMI",
    "os": "OSSETIAN",
    "bua": "BURIAT",
    "xal": "KALMYK",
    "tyv": "TUVINIAN",
    "sah": "SAKHA",
    "kaa": "KARAKALPAK",
    "ar": "ARABIC",
    "fa": "PERSIAN",
    "ug": "UYGHUR",
    "ur": "URDU",
    "ps": "PASHTO",
    "sd": "SINDHI",
    "bal": "BALOCHI",
    "hi": "HINDI",
    "mr": "MARATHI",
    "ne": "NEPALI",
    "bh": "BIHARI",
    "mai": "MAITHILI",
    "ang": "OLD ENGLISH",
    "bho": "BHOJPURI",
    "mah": "MAGAHI",
    "sck": "SADRI",
    "new": "NEWAR",
    "gom": "KONKANI",
    "sa": "SANSKRIT",
    "bgc": "HARYANVI",
    "ta": "TAMIL",
    "hu": "HUNGARIAN",
    "rs_latin": "SERBIAN LATIN",
    "id": "INDONESIAN",
    "oc": "OCCITAN",
    "is": "ICELANDIC",
    "lt": "LITHUANIAN",
    "mi": "MAORI",
    "ms": "MALAY",
    "nl": "DUTCH",
    "no": "NORWEGIAN",
    "pl": "POLISH",
    "sk": "SLOVAK",
    "sl": "SLOVENIAN",
    "sq": "ALBANIAN",
    "sv": "SWEDISH",
    "sw": "SWAHILI",
    "tl": "TAGALOG",
    "tr": "TURKISH",
    "la": "LATIN",
    "be": "BELARUSIAN",
    "th": "THAI",
    "az": "AZERBAIJANI",
    "lv": "LATVIAN",
    "pi": "PALI",
    "vi": "VIETNAMESE",
    "eu": "BASQUE",
    "lb": "LUXEMBOURGISH",
}


def debug_detection(image_path: str, detections: list, height: int = 720):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    overlay = img.copy()

    for _, poly in detections:
        pts = np.array([[pt[0], pt[1]] for pt in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))

    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    h, w = result.shape[:2]
    scale = height / h
    result = cv2.resize(result, (int(w * scale), height))

    cv2.imshow("Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_inpaint(inpainted: np.ndarray, height: int = 720):
    h, w = inpainted.shape[:2]
    scale = height / h
    resized = cv2.resize(inpainted, (int(w * scale), height))
    cv2.imshow("Inpaint", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_translation(blocks: list[dict]):
    text = json.dumps(blocks, ensure_ascii=False, indent=2)
    text = re.sub(
        r'\[\n( *-?\d+,\n)* *-?\d+\n *\]',
        lambda m: '[' + ', '.join(
            x.strip().rstrip(',') for x in m.group(0).split('\n')[1:-1]
        ) + ']',
        text,
    )
    print(text)
