import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import ocr
import llm


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <image_path> <source_lang> <target_lang>")
        sys.exit(1)

    image_path, source_lang, target_lang = sys.argv[1], sys.argv[2], sys.argv[3]

    print("Running OCR...")
    detections = ocr.detect(image_path)

    if not detections:
        print("No text detected.")
        sys.exit(0)

    print(f"Detected {len(detections)} text regions. Translating...")
    texts = [text for text, _ in detections]
    translations = llm.translate(texts, source_lang, target_lang)

    original = Image.open(image_path).convert("RGB")
    translated = original.copy()
    draw = ImageDraw.Draw(translated)

    for text, (x_min, y_min, x_max, y_max) in detections:
        translated_text = translations.get(text, text)
        box_height = max(y_max - y_min, 8)
        try:
            font = ImageFont.load_default(size=int(box_height * 0.8))
        except TypeError:
            font = ImageFont.load_default()
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")
        draw.text((x_min + 2, y_min + 2), translated_text, fill="black", font=font)

    original_cv = np.array(original)[:, :, ::-1]
    translated_cv = np.array(translated)[:, :, ::-1]
    combined = np.hstack([original_cv, translated_cv])
    cv2.imshow("Before | After", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()