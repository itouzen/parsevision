import pytesseract
from pytesseract import Output
import numpy as np
import cv2


def process(images):
    """Run Tesseract OCR on images and return (processed_images, ocr_pages)."""
    processed_images = []
    ocr_pages = []
    for page_index, image in enumerate(images):
        img = np.array(image)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        page_items = []
        n = len(data["text"])
        for i in range(n):
            try:
                conf_val = float(data["conf"][i])
            except ValueError:
                conf_val = -1.0
            text_val = data["text"][i].strip()
            if conf_val > 0 and text_val:
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                page_items.append(
                    {
                        "text": text_val,
                        "confidence": conf_val,
                        "bbox": {
                            "left": x,
                            "top": y,
                            "width": w,
                            "height": h,
                        },
                    }
                )
        processed_images.append(img)
        ocr_pages.append(
            {
                "engine": "tesseract",
                "page": page_index,
                "items": page_items,
            }
        )
    return processed_images, ocr_pages
